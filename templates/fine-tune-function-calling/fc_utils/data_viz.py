import datasets
import matplotlib.pyplot as plt
import numpy as np
import ray.data

from fc_utils.preprocessing import initial_mapper, pprint_example

hf_ds = datasets.load_dataset(
    "glaiveai/glaive-function-calling-v2", split="train"
).shuffle()
train_hf_ds = hf_ds.select(range(int(len(hf_ds) * 0.1)))
test_hf_ds = hf_ds.select(range(int(len(hf_ds) * 0.1), int(len(hf_ds) * 0.11)))
ray_ds = ray.data.from_huggingface(hf_ds)
openai_fmt_ds = ray_ds.map(initial_mapper)

pprint_example(openai_fmt_ds.take(1)[0])


def get_counts(ds):
    counts = {
        "Normal-Single": [0, 0, 0],
        "Normal-Multi": [0, 0, 0],
        "Tool-Single": [0, 0, 0],
        "Tool-Multi": [0, 0, 0],
    }
    for ex in ds.iter_rows():
        count = len(eval(ex["tools"]) if ex["tools"] else [])
        roles = [message["role"] for message in ex["messages"]]
        num_user = len([role for role in roles if role == "user"])
        is_multi = num_user > 1 if roles[-1] != "user" else num_user > 2
        if count == 0:
            if is_multi:
                counts["Normal-Multi"][0] += 1
            else:
                counts["Normal-Single"][0] += 1
        elif count == 1:
            if "tool" not in roles:
                if is_multi:
                    counts["Normal-Multi"][1] += 1
                else:
                    counts["Normal-Single"][1] += 1
            else:
                if is_multi:
                    counts["Tool-Multi"][1] += 1
                else:
                    counts["Tool-Single"][1] += 1
        else:
            if "tool" not in roles:
                if is_multi:
                    counts["Normal-Multi"][2] += 1
                else:
                    counts["Normal-Single"][2] += 1
            else:
                if is_multi:
                    counts["Tool-Multi"][2] += 1
                else:
                    counts["Tool-Single"][2] += 1
    return counts


counts = get_counts(openai_fmt_ds)


# Number of bars per group
n_groups = 3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

total_count = openai_fmt_ds.count()
normal_single = 100 * np.array(counts["Normal-Single"]) / total_count
normal_multi = 100 * np.array(counts["Normal-Multi"]) / total_count
tool_single = 100 * np.array(counts["Tool-Single"]) / total_count
tool_multi = 100 * np.array(counts["Tool-Multi"]) / total_count

# code for pie chart version
# labels = ['0 Funcs; Normal Response; Single Turn', '0 Funcs; Normal Response; Multi Turn', '1 Func; Normal Response; Single Turn', '1 Func; Normal Response; Multi Turn', '1 Func; Tool Call; Single Turn', '1 Func; Tool Call; Multi Turn', '2 Funcs; Tool Call; Single Turn', '2 Funcs; Tool Call; Multi Turn']
# sizes = [normal_single[0], normal_multi[0], normal_single[1], normal_multi[1], tool_single[1], tool_multi[1], tool_single[2], tool_multi[2]]

# fig, ax = plt.subplots()
# ax.pie(sizes, labels=labels, autopct='%1.2f%%')

rects1 = plt.bar(
    index, normal_single, bar_width, alpha=opacity, color="b", label="Normal-Single"
)
rects2 = plt.bar(
    index + bar_width,
    normal_multi,
    bar_width,
    alpha=opacity,
    color="g",
    label="Normal-Multi",
)
rects3 = plt.bar(
    index + 2 * bar_width,
    tool_single,
    bar_width,
    alpha=opacity,
    color="r",
    label="Tool-Single",
)
rects4 = plt.bar(
    index + 3 * bar_width,
    tool_multi,
    bar_width,
    alpha=opacity,
    color="y",
    label="Tool-Multi",
)


# Adding percentages on top of the bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


for rect in [rects1, rects2, rects3, rects4]:
    add_labels(rect)

# add_labels([rects1, rects2, rects3, rects4])


plt.xlabel("Number of Tools")
plt.ylabel("Percentage")
plt.title("Distribution of Responses and Turns with Number of Tools")
plt.xticks(index + bar_width * 1.5, [0, 1, 2])
plt.legend()

plt.tight_layout()
plt.show()
