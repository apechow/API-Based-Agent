import asyncio
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.append('evaluation/webarena/webarena')
sys.path.append('evaluation/webarena/webarena/evaluation_harness')
from utils import check_correctness, get_initial_prompt_from_task, get_tests

from opendevin.controller.state.state import State
from opendevin.core.config import config, get_llm_config_arg, get_parser
from opendevin.core.logger import get_console_handler
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import main
from opendevin.events.action import MessageAction


def cleanup():
    print('Cleaning up child processes...')
    for process in mp.active_children():
        print(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def codeact_user_response(state: State) -> str:
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        #'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n'
        #'HINT: use the function get_api_documentation(api) to retrieve documentation of APIs that are useful to solving the task.'
    )
    return (
        msg
        + '\nWhen you think you successfully finished the task, first respond with `Finish[answer]` where you include *only* your answer to the questionin `[]` if the user asks for an answer, make sure you should only include the answer to the question but not any additional explanation, details, or commentary unless specifically requested.'
        + '\nAfter that, when you responded with your answer, you should respond with <finish></finish>.'
        + '\nThen finally, to exit, you can run <execute_bash>\nexit()\n</execute_bash>'
    )


def monologue_user_response(state: State) -> str:
    raise NotImplementedError('MonologueAgent should never ask for user responses.')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'MonologueAgent': monologue_user_response,
    'DelegatorAgent': codeact_user_response,
    'InterleavingAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have completed the request, please run the following command: <execute_bash> exit </execute_bash>.\n'
}


def process_instance(task, agent_class, metadata, reset_logger: bool = True):
    # create process-specific workspace dir
    # we will create a workspace directory for EACH process
    # so that different agent don't interfere with each other.
    workspace_mount_path = config.workspace_mount_path
    pathlib.Path(workspace_mount_path).mkdir(parents=True, exist_ok=True)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    eval_output_dir = metadata['eval_output_dir']
    task_id = task['task_id']

    if reset_logger:
        # Set up logger
        log_file = os.path.join(eval_output_dir, 'logs', f'instance_{task_id}.log')
        # Remove all existing handlers from logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # add back the console handler to print ONE line
        logger.addHandler(get_console_handler())
        logger.info(
            f'Starting evaluation for instance {task_id}.\nHint: run "tail -f {log_file}" to see live logs in a separate shell'
        )
        # Remove all existing handlers from logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    logger.info(f'Process-specific workspace mounted at {workspace_mount_path}')

    # Prepare instruction
    instruction = get_initial_prompt_from_task(task)
    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    instruction += AGENT_CLS_TO_INST_SUFFIX.get(agent_class, '')
    logger.info(f'Instruction:\n{instruction}', extra={'msg_type': 'OBSERVATION'})

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State = asyncio.run(
        main(
            instruction,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(agent_class),
        )
    )
    if state is None:
        raise ValueError('State should not be None.')

    output_log = ''
    for i, (act, obs) in enumerate(state.history):
        output_log += f'Step {i}:\nact - {act}\nobs - {obs}\n\n'
    log_file = os.path.join(eval_output_dir, 'logs', f'instance_{task_id}.log')
    with open(log_file, 'w') as f:
        f.write(output_log)

    model_answer_raw = ''
    model_answer_last = ''
    for act, obs in state.history:
        if isinstance(act, MessageAction) and act.source == 'agent':
            if (
                act.content.strip() != ''
                and act.content.strip() != 'Too many errors encountered. Task failed.'
            ):
                model_answer_raw += f'{act.content}\n'
                model_answer_last = f'{act.content}\n'
        elif act.source == 'agent':
            if (
                act.thought.strip() != ''
                and act.thought.strip() != 'Too many errors encountered. Task failed.'
            ):
                model_answer_last = f'{act.thought}\n'
                model_answer_raw += f'{act.thought}\n'
                model_answer_raw += f'{obs}\n'
    # attempt to parse model_answer
    correct = check_correctness(task, model_answer_raw, log_file)
    metrics = state.metrics.get() if state.metrics else None
    logger.info(f'Raw Answer: {model_answer_raw} | Correct: {correct}')
    # Save the output
    output = {
        'task_id': task_id,
        'raw': model_answer_last,
        'answer_id': 'None',
        'model_id': metadata['model_name'],
        'metadata': metadata,
        #'history': [
        #    (event_to_dict(action), event_to_dict(obs)) for action, obs in state.history
        # ],
        'metrics': metrics,
        'error': state.error if state and state.error else None,
        'correct': correct,
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return output


def _replace_urls_in_obj(obj, old_url, new_url):
    """Recursively replace old_url with new_url in strings inside dicts/lists."""
    if isinstance(obj, str):
        return obj.replace(old_url, new_url)
    if isinstance(obj, dict):
        return {k: _replace_urls_in_obj(v, old_url, new_url) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_urls_in_obj(item, old_url, new_url) for item in obj]
    return obj


def process_instance_with_urls(
    task, agent_class, metadata, server_urls, orig_urls, reset_logger=True
):
    """
    Wrapper around process_instance that overrides server URLs in os.environ
    and rewrites matching URLs inside the task dict.

    server_urls: {ENV_VAR: worker_url}  — what to set in os.environ
    orig_urls:   {ENV_VAR: old_url}     — the URLs currently embedded in the task dict
                                          (built in the parent process from .env)

    The task dict is rewritten so every occurrence of old_url is replaced with
    the worker URL before process_instance sees it. This is necessary because
    prompt.py splits task['start_url'] on site_base to extract the path, so
    both must use the same host:port.
    """
    import copy

    task = copy.deepcopy(task)
    for env_key, new_url in server_urls.items():
        old_url = orig_urls.get(env_key, '')
        if old_url and new_url and old_url != new_url:
            task = _replace_urls_in_obj(task, old_url, new_url)

    saved = {k: os.environ.get(k) for k in server_urls}
    try:
        os.environ.update(server_urls)
        return process_instance(task, agent_class, metadata, reset_logger=reset_logger)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def run_multi_docker(tests, agent_class, metadata, task_timeout: int = 1800):
    """
    Run tasks in parallel, one per available docker worker.

    Each task gets its own docker worker acquired over SSH. The agent runs
    inside a subprocess (ProcessPoolExecutor) so os.environ mutations for
    GITLAB/SHOPPING/REDDIT URLs are isolated per task.

    task_timeout: seconds before a hung task is cancelled and its worker released
                  (default 30 min — set lower for faster failure detection).
    """
    from worker_pool.workers import (
        acquire_worker,
        num_workers,
        release_worker,
        server_urls_for_worker,
    )

    n = num_workers()
    logger.info(f'Multi-docker mode: {n} workers available, {len(tests)} tasks queued')

    sem = asyncio.Semaphore(n)
    acquire_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=n)

    async def run_one(task):
        async with sem:
            async with acquire_lock:
                worker = await asyncio.to_thread(acquire_worker, str(task['task_id']))

            worker_id = worker['worker_id']
            logger.info(f"Task {task['task_id']} → worker {worker_id}")
            task_failed = False
            try:
                urls = server_urls_for_worker(worker)
                # Capture current (parent-process) values so the subprocess can
                # rewrite task URLs that were built against the local .env URLs.
                orig_urls = {k: os.environ.get(k, '') for k in urls}
                fut = loop.run_in_executor(
                    executor,
                    process_instance_with_urls,
                    task,
                    agent_class,
                    metadata,
                    urls,
                    orig_urls,
                    True,
                )
                output = await asyncio.wait_for(fut, timeout=task_timeout)
            except asyncio.TimeoutError:
                task_failed = True
                logger.error(
                    f"Task {task['task_id']} timed out after {task_timeout}s on worker {worker_id}"
                )
                output = {
                    'task_id': task['task_id'],
                    'error': f'timed out after {task_timeout}s',
                    'correct': False,
                    'raw': '',
                    'worker_id': worker_id,
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            except BaseException as exc:
                task_failed = True
                import traceback as _tb

                logger.error(
                    f"Task {task['task_id']} failed on worker {worker_id}: {exc}\n"
                    + _tb.format_exc()
                )
                output = {
                    'task_id': task['task_id'],
                    'error': str(exc),
                    'correct': False,
                    'raw': '',
                    'worker_id': worker_id,
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            finally:
                release_worker(worker_id, read_only=task_failed)
                logger.info(
                    f"Worker {worker_id} released (task {task['task_id']}, read_only={task_failed})"
                )

            return output

    futures = [asyncio.ensure_future(run_one(t)) for t in tests]
    results = await asyncio.gather(*futures, return_exceptions=True)
    executor.shutdown(wait=True)
    return [r for r in results if isinstance(r, dict)]


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--start_task_id',
        type=int,
        help='which task_id to start from',
        default=132,
    )
    parser.add_argument(
        '--multi-docker',
        action='store_true',
        default=False,
        help=(
            'Run tasks in parallel against the multi-docker worker pool via SSH. '
            'Requires REMOTE_HOST env var (e.g. user@host). '
            'Each task acquires an isolated docker worker so evaluations do not interfere.'
        ),
    )
    args, _ = parser.parse_known_args()
    if args.directory:
        config.workspace_base = os.path.abspath(args.directory)
        print(f'Setting workspace base to {config.workspace_base}')
    # Check https://github.com/OpenDevin/OpenDevin/blob/main/evaluation/swe_bench/README.md#configure-opendevin-and-your-llm
    # for details of how to set `llm_config`
    if args.llm_config:
        specified_llm_config = get_llm_config_arg(args.llm_config)
        if specified_llm_config:
            config.llm = specified_llm_config
    logger.info(f'Config for evaluation: {config}')
    agent_class = args.agent_cls
    assert (
        agent_class in AGENT_CLS_TO_FAKE_USER_RESPONSE_FN
    ), f'Unsupported agent class: {agent_class}'
    model_name = config.llm.model.split('/')[-1]
    max_iterations = args.max_iterations
    eval_note = ''
    if args.eval_note is not None:
        eval_note += '_N_' + args.eval_note
    eval_output_dir = os.path.join(
        args.eval_output_dir,
        agent_class,
        model_name + '_maxiter_' + str(max_iterations) + eval_note,
    )
    pathlib.Path(eval_output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(eval_output_dir, 'logs')).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f'Using evaluation output directory: {eval_output_dir}')
    logger.info('Evaluating Webarena tests')
    workspace_mount_path = config.workspace_mount_path
    pathlib.Path(workspace_mount_path).mkdir(parents=True, exist_ok=True)
    tests = get_tests(args.start_task_id)
    print(f'len(tests) is {len(tests)}')
    # TEST METADATA
    metadata = {
        'agent_class': agent_class,
        'model_name': model_name,
        'max_iterations': max_iterations,
        'eval_output_dir': eval_output_dir,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        # get the commit id of current repo for reproduciblity
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip(),
    }
    logger.info(f'Metadata: {metadata}')
    with open(os.path.join(eval_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    # LIMIT EVALUATION
    eval_n_limit = args.eval_n_limit
    if eval_n_limit:
        tests = tests[:eval_n_limit]
        logger.info(
            f'Limiting evaluation to a total of first {eval_n_limit} instances. ({len(tests)})'
        )
    if tests[0]['sites'] == ['reddit', 'gitlab'] or tests[0]['sites'] == [
        'gitlab',
        'reddit',
    ]:
        output_file = os.path.join(
            eval_output_dir, f'output_gitlab_reddit_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['map', 'wikipedia'] or tests[0]['sites'] == [
        'wikipedia',
        'map',
    ]:
        output_file = os.path.join(
            eval_output_dir, f'output_map_wikipedia_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['map', 'shopping_admin'] or tests[0]['sites'] == [
        'shopping_admin',
        'map',
    ]:
        output_file = os.path.join(
            eval_output_dir, f'output_map_shopping_admin_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['gitlab', 'wikipedia'] or tests[0]['sites'] == [
        'wikipedia',
        'gitlab',
    ]:
        output_file = os.path.join(
            eval_output_dir, f'output_gitlab_wikipedia_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['shopping', 'reddit'] or tests[0]['sites'] == [
        'shopping',
        'reddit',
    ]:
        output_file = os.path.join(
            eval_output_dir, f'output_shopping_reddit_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['shopping_admin']:
        output_file = os.path.join(
            eval_output_dir, f'output_shopping_admin_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['shopping']:
        output_file = os.path.join(
            eval_output_dir, f'output_shopping_{model_name}.jsonl'
        )
    elif tests[0]['sites'] == ['gitlab']:
        output_file = os.path.join(eval_output_dir, f'output_gitlab_{model_name}.jsonl')
    elif tests[0]['sites'] == ['reddit']:
        output_file = os.path.join(eval_output_dir, f'output_reddit_{model_name}.jsonl')
    elif tests[0]['sites'] == ['map']:
        output_file = os.path.join(eval_output_dir, f'output_map_{model_name}.jsonl')
    else:
        assert 1 == 2
        output_file = os.path.join(eval_output_dir, f'output_{model_name}.jsonl')
    logger.info(f'Writing evaluation output to {output_file}')
    finished_task_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                task = json.loads(line)
                finished_task_ids.add(task['task_id'])
        logger.warning(
            f'Output file {output_file} already exists. Loaded {len(finished_task_ids)} finished instances.'
        )
    output_fp = open(output_file, 'a')
    logger.info(
        f'Evaluation started with Agent {agent_class}, model {model_name}, max iterations {max_iterations}.'
    )
    # =============================================
    # filter out finished instances
    new_tests = []
    for task in tests:
        task_id = task['task_id']
        if task_id in finished_task_ids:
            logger.info(f'Skipping instance {task_id} as it is already finished.')
            continue
        new_tests.append(task)
    finished_task_number = len(finished_task_ids)
    tests = new_tests
    logger.info(
        f'Finished instances: {finished_task_number}, Remaining instances: {len(tests)}'
    )

    # =============================================
    # Multi-docker parallel path
    if args.multi_docker:
        logger.info(
            'Multi-docker mode enabled — running tasks in parallel via SSH worker pool'
        )
        outputs = asyncio.run(run_multi_docker(tests, agent_class, metadata))
        for output in outputs:
            logger.info(
                f'Finished evaluation for instance {output["task_id"]}: '
                f'correctness - {output.get("correct")}; answer - {output.get("raw", "")}'
            )
            output_fp.write(json.dumps(output) + '\n')
            output_fp.flush()
        output_fp.close()
        sys.exit(0)

    # =============================================
    # Sequential (single-docker) path
    pbar = tqdm(total=len(tests))
    num_workers = args.eval_num_workers
    logger.info(f'Using {num_workers} workers for evaluation.')
    for task in tests:
        if True:
            pbar.update(1)
            output = process_instance(
                task,
                agent_class,
                metadata,
                reset_logger=bool(num_workers > 1),
            )
            pbar.set_description(f'Instance {output["task_id"]}')
            logger.info(
                f'Finished evaluation for instance {output["task_id"]}: correctness - {output["correct"]}; answer - {output["raw"]}'
            )
            output_fp.write(json.dumps(output) + '\n')
            output_fp.flush()
            finished_task_ids.add(output['task_id'])
    output_fp.close()
