import pandas as pd
import matplotlib.pyplot as plt

# ==== Step 1: Load Data ====
df = pd.read_csv("job_language.csv")  # <-- adjust if needed


# ==== Step 3: In-demand Positions (Percentage) ====
job_counts = df["IT_job_criteria"].value_counts(normalize=True) * 100
job_counts = job_counts.reset_index()
job_counts.columns = ["Job", "Percentage"]

# Save percentages
job_counts.to_csv("job_percentages.csv", index=False)

# Plot top 10 jobs
plt.figure(figsize=(10,6))
job_counts.head(10).plot(kind="bar", x="Job", y="Percentage", legend=False)
plt.title("Top 10 In-demand Positions")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top10_jobs.png")
plt.close()

# Plot top 5 jobs
plt.figure(figsize=(10,6))
job_counts.head(5).plot(kind="bar", x="Job", y="Percentage", legend=False)
plt.title("Top 5 In-demand Positions")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top5_jobs.png")
plt.close()

# ==== Step 4: In-demand Programming Languages (Percentage) ====
lang_columns = [col for col in df.columns if "programing_language" in col.lower()]
print("Detected language columns:", lang_columns)

if not lang_columns:
    print("⚠️ No language columns found in dataset. Please check column names.")
else:
    # Stack all language columns
    languages = df[lang_columns].melt(value_name="Language")["Language"].dropna()
    languages = languages.str.strip()

    # Group Web-related labels
    languages = languages.replace({
        "JS": "Web-related",
        "HTML": "Web-related",
        "CSS": "Web-related"
    })

    # Count percentages
    lang_counts = languages.value_counts(normalize=True) * 100
    lang_counts = lang_counts.reset_index()
    lang_counts.columns = ["Language", "Percentage"]

    # Save percentages
    lang_counts.to_csv("language_percentages.csv", index=False)

    # Plot top 8 languages
    plt.figure(figsize=(10,6))
    lang_counts.head(8).plot(kind="bar", x="Language", y="Percentage", legend=False)
    plt.title("Top 8 In-demand Programming Languages")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top8_languages.png")
    plt.close()

    # Plot top 5 languages
    plt.figure(figsize=(10,6))
    lang_counts.head(5).plot(kind="bar", x="Language", y="Percentage", legend=False)
    plt.title("Top 5 In-demand Programming Languages")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top5_languages.png")
    plt.close()

print("✅ Analysis complete! Files saved:")
print("- job_counts.csv (percentages)")
print("- language_percentages.csv (percentages)")
print("- top8_jobs.png, top5_jobs.png")
print("- top8_languages.png, top5_languages.png")
