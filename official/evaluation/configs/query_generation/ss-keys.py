from compaction.query_generation.config import QueryConfig, QueryMethodConfig, SelfStudyConfig
from compaction.query_generation.conversation_specs import repeat_specs

# Self-study ONLY, single task-targeted prompt: the collect_keys spec
# ("List every special keyword ... and the exact value assigned to it").
config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='self_study',
            fraction=1.0,
            config=SelfStudyConfig(
                conversation_specs=repeat_specs([
                    ("collect_keys", 1),
                ]),
            )
        ),
    ],
    max_query_vectors_per_kv_head=50000,
    verbose=True
)
