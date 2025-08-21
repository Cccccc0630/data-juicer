import threading
import time
from collections import defaultdict
import logging

from data_juicer.ops.base_op import Filter

# 假设 Filter 和 Mapper 类已定义
# from your_module import Filter, Mapper

logger = logging.getLogger(__name__)

### NEW CLASS: PipelineOrchestrator ###
class PipelineOrchestrator:
    """
    管理数据处理流水线的动态路由和操作员（OP）排序。
    """
    def __init__(self, operators, reorder_batch_size=1000):
        """
        初始化编排器。
        
        Args:
            operators (list): 初始的操作员列表。
            reorder_batch_size (int): 处理多少条数据后触发一次重新排序。
        """
        self.operators = operators
        self.reorder_batch_size = reorder_batch_size
        
        self.op_stats = defaultdict(lambda: {'in': 0, 'out': 0, 'time': 0.0})
        self.processed_item_counter = 0
        self.routing_table = {}
        self.filter_ops = [op for op in self.operators if isinstance(op, Filter)]
        self.non_filter_ops = [op for op in self.operators if not isinstance(op, Filter)]
        
        self._lock = threading.Lock()
        self._initialize_routing()
        
        logger.info(f"Orchestrator initialized. Reordering will trigger every {reorder_batch_size} items.")

    def _initialize_routing(self):
        """根据初始顺序设置路由表。"""
        op_names = [op._name for op in self.operators]
        
        for i, name in enumerate(op_names[:-1]):
            self.routing_table[name] = op_names[i+1]
        self.routing_table[op_names[-1]] = None # 最后一个OP没有下游
        logger.info(f"Initial routing table set: {self.routing_table}")

    def report_stats(self, op_name, items_in, items_out, time_spent):
        """
        Actor调用此方法来报告其处理统计信息。
        """
        with self._lock:
            # 仅在数据首次进入流水线时（即第一个OP报告时）增加全局计数器
            first_op_name = self.operators[0]._name
            if op_name == first_op_name:
                self.processed_item_counter += items_in

            # 累积统计数据
            self.op_stats[op_name]['in'] += items_in
            self.op_stats[op_name]['out'] += items_out
            self.op_stats[op_name]['time'] += time_spent
            
            # 检查是否需要重新排序
            if self.processed_item_counter >= self.reorder_batch_size:
                logger.info(f"Processed {self.processed_item_counter} items, triggering re-evaluation of Filter OP order.")
                self._reorder_filters()
                self.processed_item_counter = 0 # 重置计数器

    def _reorder_filters(self):
        """
        根据性能指标对Filter OP进行重新排序并更新路由表。
        此方法必须在锁内调用。
        """
        if not self.filter_ops:
            return

        perf_scores = {}
        for op in self.filter_ops:
            stats = self.op_stats[op._name]
            if stats['in'] == 0:
                score = -1 # 没有数据流过，无法评估
                continue
            
            # 计算丢弃率和平均处理时间
            drop_rate = (stats['in'] - stats['out']) / stats['in']
            avg_time_per_item = stats['time'] / stats['in']
            
            # 核心评分逻辑：我们想要高丢弃率和低处理时间。
            # score = drop_rate / avg_time_per_item  (避免除以零)
            # 为了数值稳定性，可以添加一个小的epsilon。
            score = drop_rate / (avg_time_per_item + 1e-9)
            perf_scores[op._name] = score
            
            logger.info(f"Filter OP '{op._name}': Drop Rate={drop_rate:.2%}, Avg Time={avg_time_per_item:.4f}s, Score={score:.2f}")

        # 根据分数降序排序Filter OP
        sorted_filter_names = sorted(self.filter_ops, key=lambda op: perf_scores.get(op._name, -1), reverse=True)
        sorted_filter_names = [op._name for op in sorted_filter_names]
        
        # 将非Filter OP插入回其原始相对位置（简化策略）
        # 这里我们假设所有Filter OP是连续的，并将它们作为一个整体进行排序
        # 一个更稳健的策略是将所有OP链接起来
        
        # 构建新的完整操作链
        # 假设所有非Filter OP保持其原始相对顺序
        new_op_order = []
        original_op_names = [op._name for op in self.operators]
        sorted_filters_iter = iter(sorted_filter_names)

        # 重新构建完整的OP链
        for name in original_op_names:
            if name in perf_scores: # 如果是Filter OP
                # 如果它还没被添加，就添加整个排序后的Filter OP块
                if sorted_filter_names:
                    new_op_order.extend(sorted_filter_names)
                    sorted_filter_names = [] # 清空以防重复添加
            else: # 如果是非Filter OP
                new_op_order.append(name)

        # 更新路由表
        new_routing_table = {}
        for i, name in enumerate(new_op_order[:-1]):
            new_routing_table[name] = new_op_order[i+1]
        new_routing_table[new_op_order[-1]] = None # 最后一个OP
        
        self.routing_table = new_routing_table
        logger.warning(f"Pipeline order re-evaluated. New routing: {self.routing_table}")

    def get_next_op_name(self, current_op_name):
        """获取指定OP的下一个目标OP名称。"""
        with self._lock:
            return self.routing_table.get(current_op_name)