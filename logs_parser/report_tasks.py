try:
    from report_utils.tasks import print_task_summary
    from report_utils.models import print_failed_task_model_stats
    from report_utils.codegen import print_codegen_analysis
except ImportError:
    from logs_parser.report_utils.tasks import print_task_summary
    from logs_parser.report_utils.models import print_failed_task_model_stats
    from logs_parser.report_utils.codegen import print_codegen_analysis

# This file now acts as a facade.
