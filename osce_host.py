from grade_osce import evaluate_transcript, get_client_from_env, intuit_student_and_case
from pathlib import Path
import json 










def main():
    fails = 0
    transcripts_dir = "student_transcripts"
    name = input("Please enter your name: ")
    print(f"Hello, {name}! Welcome to the OSCE Grading System.")
    while fails < 3:
        student = input("Do you know the student ID you would like to evaluate? (y/n): ").strip().lower()
        if student != 'y':
            print("Let's help you find the student ID.")
            transcripts = sorted(Path(transcripts_dir).glob("*.txt"))
            student_ids = [intuit_student_and_case(file)[0] for file in transcripts]
            print("Available student IDs:")
            for sid in sorted(set(student_ids)):
                print(f"    - {sid}")
        request = input("Enter the student ID you would like to evaluate: ").strip().lower()
        files = sorted(Path(transcripts_dir).glob(f"{request}*.txt"))
        result_list = []
        if not files:
            print(f"No transcript files found for student ID: {request}")
            print("Please check the ID (student name) and try again.")
            fails +=1
            if fails == 3:
                print("Maximum number of failed attempts reached. Exiting the program.")
                return
            continue
        for file in files:
            print(f"Evaluating transcript file: {file.name}")
            client = get_client_from_env()
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0"
            print(f"Using model: {model}")
            result_list.append(evaluate_transcript(
                file, 
                client=client, 
                model=model, 
                retry=0, 
                MAX_RETRIES=5
            ))
        output_file = Path("raw_scores_summaries") / f"{request}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result_list, f, indent=4)
        print(f"Results saved to {output_file}")
        json_str = json.dumps(result_list, indent=2)
        prompt = f"""
            You are an AI teaching assistant helping a medical educator.
            Here is JSON output from a tool called "grade_osce" that 
            evaluated student {request} across several OSCE encounters:

            {json_str}

            Please generate a short, 1-2 paragraph narrative summary of this 
            student's performance,
            highlighting strengths and areas for improvement in:
            - History & Information Gathering,
            - Diagnostic Reasoning,
            - Empathy & Rapport,
            - Information Giving & Clarity,
            - Organization & Closure.
            """
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                # Removed response_format - causes issues with Claude via LiteLLM
                # Rely on system prompt instructions for JSON output instead
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            print(f"ERROR: error {e} occurred.")
            continue

        content = resp.choices[0].message.content
        print(f"AI Narrative Summary for student {request}:\n{content}\n")
        write_ai_summary = input("Would you like to save this summary to a file? (y/n): ").strip().lower()
        if write_ai_summary == 'y':
            summary_file = Path("narrative_summaries") / f"{request}.txt"
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_file, "w") as f:
                f.write(content)
            print(f"Narrative summary saved to {summary_file}")
        another = input("would you like to evaluate another student? (y/n): ").strip().lower()
        if another != 'y':
            print("Exiting the OSCE Grading System. Goodbye!")
            break
        else:
            fails = 0  # reset fails on successful attempt


if __name__ == "__main__":
    main()
