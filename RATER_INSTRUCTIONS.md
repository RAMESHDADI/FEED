# Instructions for raters (give this page + the CSV sheet to each rater)

You will see a question, a student's answer and a piece of feedback written for that answer.
Work alone. Do not discuss items with the other rater. Do not search for the "right" rating together.

## Audit sheet (audit_sheet.csv) - fill 3 columns for every row
1. **factually_correct_0or1** - 1 if every technical statement in the feedback is correct for an undergraduate Operating Systems course; 0 if anything in it is wrong or misleading.
2. **pedagogically_useful_1to5** - 1 = useless or harmful, 3 = acceptable but generic, 5 = tells the student exactly what is right/missing and how to improve.
3. **problem_type** - leave empty if fine; otherwise one word: `factual` (wrong content), `vague` (generic), `mismatch` (does not match the student's answer), `other`.
(Optional) corrected_feedback_optional - a better feedback if you would change it.

## Blinded comparison sheet (human_eval_sheet_*.csv) - fill 4 columns for every row
You see two feedbacks, A and B, for the same answer. You are NOT told which system wrote them.
Rate each from 1 to 5: `A_correct_1to5`, `A_useful_1to5`, `B_correct_1to5`, `B_useful_1to5`
(correct: 1 = mostly wrong, 5 = fully correct; useful: as above). Do not look at the reference feedback unless you need it to judge correctness.

Time needed: about 2-3 minutes per row, 200 rows for the audit, 100 rows for the comparison.
Save your file as rater1.csv / rater2.csv (your own number). Send the CSV back, do not change column names or item ids.
