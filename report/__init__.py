"""The stages that turn saved predictions into the paper's tables and figures.

Nothing here fits a model: each module reads what the pipeline wrote and
reduces it. Run them as modules from the repository root, for example
`python -m report.make_tables --tables 1 6`.
"""
