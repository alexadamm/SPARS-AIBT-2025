from SPARS.Simulator.Algo.easy_auto_switch_on import EASYAuto
from SPARS.Simulator.Algo.easy_normal import EASYNormal
from SPARS.Simulator.Algo.fcfs_auto_switch_on import FCFSAuto
from SPARS.Simulator.Algo.fcfs_normal import FCFSNormal
from SPARS.Simulator.Algo.easy_mcf import EASYMinimumCostFlow

ALGO_MAP = {
    'fcfs_auto': FCFSAuto,
    'fcfs_normal': FCFSNormal,
    'easy_normal': EASYNormal,
    'easy_auto': EASYAuto,
    'easy_mcf': EASYMinimumCostFlow,
}


class Scheduler:
    def __init__(self, state, waiting_queue, algorithm, start_time, jobs_manager, timeout=None, platform_info=None, workload_info=None):
        AlgorithmClass = ALGO_MAP[algorithm.lower()]
        if AlgorithmClass == EASYMinimumCostFlow:
            if platform_info is None or workload_info is None:
                raise ValueError(
                    "platform_info and workload_info must be provided for EASYMinimumCostFlow algorithm")
            self.algorithm = AlgorithmClass(
                state,
                waiting_queue,
                start_time,
                jobs_manager,
                timeout,
                platform_info=platform_info,
                workload_info=workload_info
            )
        else:
            self.algorithm = AlgorithmClass(
                state,
                waiting_queue,
                start_time,
                jobs_manager,
                timeout
            )
        self.jobs_manager = jobs_manager

    def schedule(self, current_time, new_state, waiting_queue, scheduled_queue, resources_agenda):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule(
            new_state, waiting_queue, scheduled_queue, resources_agenda)
        return events
