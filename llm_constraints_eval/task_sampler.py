import itertools

class TaskSampler:
    def __init__(self, tasks):
        self.tasks = tasks  # tasks is a dict with task IDs as keys

    def sample_constraints(self, task_id, num_types):
        """
        Generate all possible combinations of constraints for a task based on the number of types.
        Returns combinations of constraint IDs.
        """
        task = self.tasks[task_id]

        constraint_types = ['skill_constraints', 'item_constraints', 'environment_constraints']
        constraints = {ctype: task.get(ctype, []) for ctype in constraint_types}

        # Remove types with empty constraints
        available_types = [ctype for ctype in constraint_types if constraints[ctype]]

        # If the number of available types is less than num_types, return empty list
        if len(available_types) < num_types:
            return []

        # Generate combinations of constraint types
        type_combinations = list(itertools.combinations(available_types, num_types))

        all_combinations = []
        for type_combo in type_combinations:
            # Get all constraints for the selected types
            constraint_lists = []
            constraint_ids_lists = []
            for ctype in type_combo:
                constraint_texts = constraints[ctype]
                constraint_ids = list(range(len(constraint_texts)))  # Use indices as IDs
                constraint_lists.append(constraint_texts)
                constraint_ids_lists.append(constraint_ids)

            # Generate the product of constraints across selected types
            for constraint_values, constraint_ids in zip(
                itertools.product(*constraint_lists),
                itertools.product(*constraint_ids_lists)
            ):
                all_combinations.append({
                    'constraint_types': type_combo,
                    'constraint_texts': constraint_values,
                    'constraint_ids': constraint_ids
                })
        return all_combinations
    

def generate_query(task_description, constraints, time_constraint, template):
    """
    Generate the query by filling in the placeholders in the template.
    """
    constraints_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(constraints)])
    query = template.replace("<TASK>", task_description)
    query = query.replace("<CS>", constraints_str)
    query = query.replace("<TC>", time_constraint)
    return query