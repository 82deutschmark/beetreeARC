import time
try:
    from rich.table import Table
    from rich import box
except ImportError:
    pass

STATUS_ICON = {
    "RUNNING": "🟡",
    "COMPLETED": "🟢",
    "ERROR": "🔴",
    "WARNING": "🟠",
}

def render_table(task_states):
    table = Table(box=box.SIMPLE)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Step", style="magenta")
    table.add_column("Outcome", style="bold")
    table.add_column("Duration", justify="right")

    # Sort by status (Running first), then task ID
    def sort_key(item):
        key, state = item
        status_order = {"RUNNING": 0, "WARNING": 1, "ERROR": 2, "COMPLETED": 3}
        return (status_order.get(state.get("status", "RUNNING"), 4), key)

    for key, state in sorted(task_states.items(), key=sort_key):
        task_str = f"{state['task_id']}:{state['test_index']}"
        status = state.get('status', 'UNKNOWN')
        icon = STATUS_ICON.get(status, "⚪")
        
        step = state.get('step', '-')
        outcome = state.get('outcome', '-')
        if outcome == "PASS":
            outcome_style = "[green]PASS[/green]"
        elif outcome == "FAIL":
            outcome_style = "[red]FAIL[/red]"
        else:
            outcome_style = outcome

        start_time = state.get('start_time')
        duration_str = "-"
        if start_time:
            end_time = state.get('end_time') or time.time()
            duration = end_time - start_time
            minutes, seconds = divmod(int(duration), 60)
            duration_str = f"{minutes:02d}:{seconds:02d}"

        table.add_row(
            task_str,
            f"{icon} {status}",
            step,
            outcome_style,
            duration_str
        )
    return table

def update_task_states(task_states, msg):
    key = f"{msg['task_id']}:{msg['test_index']}"
    if key not in task_states:
        task_states[key] = {}
    
    state = task_states[key]
    state.update(msg)
    
    # Handle WARNING specifically to persist status until next valid update
    if msg.get("event") == "WARNING":
        state["status"] = "WARNING"
        state["step"] = msg.get("step") # Contains the warning message

    if msg.get("event") == "START" and "start_time" not in state:
        state["start_time"] = msg["timestamp"]
    
    if msg.get("event") == "FINISH":
        state["end_time"] = msg["timestamp"]
        if "predictions" in msg:
            state["predictions"] = msg["predictions"]
