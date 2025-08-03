import json
import os
import glob
from collections import defaultdict, Counter

RESULTS_DIR = "./results"  

def load_results(results_dir):
    files = glob.glob(os.path.join(results_dir, "*.json"))
    all_data = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            data["exp_name"] = data.get("exp_name", os.path.basename(file))
            all_data.append(data)
    return all_data

def aggregate_final_results(all_data):
    total_questions = 0
    weighted_accuracy_sum = 0
    per_agent_accuracy_sum = defaultdict(float)
    per_agent_accuracy_weight = defaultdict(int)
    agreement_sum = 0
    avg_round_sum = 0
    round_histogram = Counter()
    per_agent_changes_agg = defaultdict(lambda: {"changed_total": 0, "changed_to_correct": 0, "changed_to_wrong": 0})
    retrieval_calls_sum = 0
    retrieval_used_sum = 0
    questions_all_agents_failed = set()
    questions_all_agents_agreed = set()
    per_experiment_accuracy = {}
    global_agent_failures = defaultdict(int)
    experiments_count = len(all_data)

    for exp in all_data:
        exp_name = exp["exp_name"]
        total_q = exp["total_questions"]
        exp_acc = exp["overall_final_accuracy"]
        per_experiment_accuracy[exp_name] = exp_acc
        failed_qs = exp.get("questions_all_agents_failed", [])
        for qid in failed_qs:
            global_agent_failures[qid] += 1


        total_questions += total_q
        weighted_accuracy_sum += exp_acc * total_q
        agreement_sum += exp["agreement_rate"] * total_q
        avg_round_sum += exp["average_round_count"] * total_q
        
        for k, v in exp["round_count_histogram"].items():
            round_histogram[k] += v
        
        for agent, acc in exp["per_agent_accuracy"].items():
            per_agent_accuracy_sum[agent] += acc * total_q
            per_agent_accuracy_weight[agent] += total_q
        
        for agent, changes in exp["per_agent_changes"].items():
            per_agent_changes_agg[agent]["changed_total"] += changes["changed_total"]
            per_agent_changes_agg[agent]["changed_to_correct"] += changes["changed_to_correct"]
            per_agent_changes_agg[agent]["changed_to_wrong"] += changes["changed_to_wrong"]

        retrieval_calls_sum += exp["retrieval"]["average_calls_per_question"] * total_q
        retrieval_used_sum += exp["retrieval"]["questions_with_retrieval_used"]

        questions_all_agents_failed.update(exp.get("questions_all_agents_failed", []))
        questions_all_agents_agreed.update(exp.get("questions_all_agents_agreed", []))
    
    overall_accuracy = weighted_accuracy_sum / total_questions
    per_agent_accuracy = {
        agent: per_agent_accuracy_sum[agent] / per_agent_accuracy_weight[agent]
        for agent in per_agent_accuracy_sum
    }
    globally_failed_questions = [
        qid for qid, count in global_agent_failures.items()
        if count == experiments_count
    ]
    best_agent = max(per_agent_accuracy.items(), key=lambda x: x[1])
    best_experiment = max(per_experiment_accuracy.items(), key=lambda x: x[1])
    
    summary = {
        "total_questions": total_questions,
        "overall_accuracy": overall_accuracy,
        "per_agent_accuracy": per_agent_accuracy,
        "best_agent": best_agent,
        "agreement_rate": agreement_sum / total_questions,
        "average_round_count": avg_round_sum / total_questions,
        "round_count_histogram": dict(round_histogram),
        "per_agent_changes": dict(per_agent_changes_agg),
        "retrieval": {
            "average_calls_per_question": retrieval_calls_sum / total_questions,
            "total_questions_with_retrieval": retrieval_used_sum
        },
        "questions_all_agents_failed": list(questions_all_agents_failed),
        "questions_all_agents_agreed": list(questions_all_agents_agreed),
        "per_experiment_accuracy": per_experiment_accuracy,
        "best_experiment": best_experiment,
        "globally_failed_questions": globally_failed_questions

    }
    return summary

def generate_analysis_report(summary):
    report = []
    report.append("### Aggregated Experiment Analysis ###\n")
    report.append(f"Total questions across experiments: {summary['total_questions']}")
    report.append(f"Overall final accuracy: {summary['overall_accuracy']:.2%}\n")

    report.append("#### Accuracy per Experiment:")
    for name, acc in summary['per_experiment_accuracy'].items():
        report.append(f" - {name}: {acc:.2%}")
    
    exp_name, exp_acc = summary['best_experiment']
    report.append(f"\nBest performing experiment: **{exp_name}** with accuracy {exp_acc:.2%}\n")
    
    report.append("#### Per-agent accuracy:")
    for agent, acc in summary['per_agent_accuracy'].items():
        report.append(f" - {agent}: {acc:.2%}")
    
    agent, acc = summary['best_agent']
    report.append(f"\nBest performing agent: **{agent}** with accuracy {acc:.2%}\n")
    
    report.append(f"Agreement rate across agents: {summary['agreement_rate']:.2%}")
    report.append(f"Average reasoning rounds: {summary['average_round_count']:.2f}")
    report.append("Round count distribution:")
    for k, v in summary['round_count_histogram'].items():
        report.append(f" - {k} rounds: {v} questions")
    
    report.append("\n#### Per-agent changes:")
    for agent, changes in summary['per_agent_changes'].items():
        report.append(
            f" - {agent}: Changed {changes['changed_total']} times "
            f"(→ correct: {changes['changed_to_correct']}, → wrong: {changes['changed_to_wrong']})"
        )
    
    report.append("\n#### Retrieval stats:")
    report.append(f" - Avg calls per question: {summary['retrieval']['average_calls_per_question']:.2f}")
    report.append(f" - Total questions with retrieval: {summary['retrieval']['total_questions_with_retrieval']}")
    
    report.append("\n#### Problematic questions:")
    report.append(f" - All agents failed (combined): {len(summary['questions_all_agents_failed'])}")
    report.append(f" - Globally failed questions (all agents wrong in all experiments): "
                  f"{len(summary['globally_failed_questions'])} → "
                  f"IDs: {summary['globally_failed_questions'][:10]}")
    report.append(f" - All agents agreed: {len(summary['questions_all_agents_agreed'])}\n")
    
    report.append("**Insights:**")
    if summary['agreement_rate'] < 0.9:
        report.append("- Agents often disagree; consider stronger coordination.")
    if len(summary['globally_failed_questions']) > 0:
        report.append("- Some questions were consistently challenging across all experiments.")
    if any(ch['changed_to_correct'] > ch['changed_to_wrong'] for ch in summary['per_agent_changes'].values()):
        report.append("- Agent changes usually improved answers, indicating beneficial revision cycles.")
    
    return "\n".join(report)

    report = []
    report.append("### Aggregated Experiment Analysis ###\n")
    report.append(f"Total questions across experiments: {summary['total_questions']}")
    report.append(f"Overall final accuracy: {summary['overall_accuracy']:.2%}\n")

    report.append("#### Accuracy per Experiment:")
    for name, acc in summary['per_experiment_accuracy'].items():
        report.append(f" - {name}: {acc:.2%}")
    
    exp_name, exp_acc = summary['best_experiment']
    report.append(f"\nBest performing experiment: **{exp_name}** with accuracy {exp_acc:.2%}\n")
    
    report.append("#### Per-agent accuracy:")
    for agent, acc in summary['per_agent_accuracy'].items():
        report.append(f" - {agent}: {acc:.2%}")
    
    agent, acc = summary['best_agent']
    report.append(f"\nBest performing agent: **{agent}** with accuracy {acc:.2%}\n")
    
    report.append(f"Agreement rate across agents: {summary['agreement_rate']:.2%}")
    report.append(f"Average reasoning rounds: {summary['average_round_count']:.2f}")
    report.append("Round count distribution:")
    for k, v in summary['round_count_histogram'].items():
        report.append(f" - {k} rounds: {v} questions")
    
    report.append("\n#### Per-agent changes:")
    for agent, changes in summary['per_agent_changes'].items():
        report.append(
            f" - {agent}: Changed {changes['changed_total']} times "
            f"(→ correct: {changes['changed_to_correct']}, → wrong: {changes['changed_to_wrong']})"
        )
    
    report.append("\n#### Retrieval stats:")
    report.append(f" - Avg calls per question: {summary['retrieval']['average_calls_per_question']:.2f}")
    report.append(f" - Total questions with retrieval: {summary['retrieval']['total_questions_with_retrieval']}")
    
    report.append("\n#### Problematic questions:")
    report.append(f" - All agents failed: {len(summary['questions_all_agents_failed'])}")
    report.append(f" - Globally failed questions (all agents wrong in all experiments): {len(summary['globally_failed_questions'])}")

    report.append(f" - All agents agreed: {len(summary['questions_all_agents_agreed'])}\n")
    
    report.append("**Insights:**")
    if summary['agreement_rate'] < 0.9:
        report.append("- Agents often disagree; consider stronger coordination.")
    if len(summary['questions_all_agents_failed']) > 0:
        report.append("- Some questions were difficult for all agents; need improved reasoning.")
    if any(ch['changed_to_correct'] > ch['changed_to_wrong'] for ch in summary['per_agent_changes'].values()):
        report.append("- Agent changes usually improved answers, indicating beneficial revision cycles.")
    
    return "\n".join(report)

if __name__ == "__main__":
    data = load_results(RESULTS_DIR)
    summary = aggregate_final_results(data)
    report = generate_analysis_report(summary)
    print(report)