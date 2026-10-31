# Social media engagement analysis

Contains cleaned spreadsheet data, presentations and linear regression code predicting engagement rate across platforms.

## Files

Start with `Models/Linear_regression_model.md`. Existing notebooks retain the original project work.

## Running the work

Use Python with pandas, NumPy, scikit-learn, Matplotlib, Jupyter and openpyxl for the data notebooks; some also use seaborn or odfpy. Open notebooks from their repository root so relative data paths resolve. Check all required input files before execution. Dependencies are not yet version-pinned; results may vary across environments. For the Manim presentation, use its original rendering workflow instead.

## Results and limitations

The published model is continuous regression, not a high/low logistic classifier. Likes, comments and other realised engagement measures are unsuitable inputs for a pre-publication forecast. Label encoding imposes arbitrary numeric order on categories. Model recommendations are exploratory, not causal.

## Contributions

For group coursework, the repository contains team work. Individual responsibilities are not documented here; do not infer sole authorship.
