from compaction.query_generation.config import QueryConfig, QueryMethodConfig, SelfStudyConfig
from compaction.query_generation.conversation_specs import repeat_specs

# Repeat-prefill backbone + ONLY the generic structure_json prompt
# (repeat + json), i.e. ss-plus-repeat trimmed to a single self-study spec.
config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='self_study',
            fraction=1.0,
            config=SelfStudyConfig(
                conversation_specs=repeat_specs([
                    ("repeat", 1),
                    ("structure_json", 1),
                ]),
            )
        ),
    ],
    max_query_vectors_per_kv_head=50000,
    verbose=True
)
