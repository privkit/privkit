# Contributing guidelines

Thanks for using Privkit and for considering contributing to it.

This toolkit aims at standardizing the privacy analysis by providing privacy-preserving mechanisms, attacks, and metrics. It is an open-source project that can be easily extended as follows.

### To extend Privkit

This toolkit follows a modular programming approach, where each module has an `Abstract Class` that defines the required methods/variables. Thus, to extend Privkit:

- Identify the module that you want to extend.
- Ensure that your code meets the requirements of the corresponding `Abstract Class`, specifically:
  - [DataType](https://github.com/privkit/privkit/blob/main/privkit/data/data_type.py)
  - [Dataset](https://github.com/privkit/privkit/blob/main/privkit/datasets/dataset.py)
  - [PPM](https://github.com/privkit/privkit/blob/main/privkit/ppms/ppm.py)
  - [Attack](https://github.com/privkit/privkit/blob/main/privkit/attacks/attack.py)
  - [Metric](https://github.com/privkit/privkit/blob/main/privkit/metrics/metric.py)
- Document your source code.

### To propose a new feature

- Check if there is any issue related to your proposal already created on the [issue tracker](https://github.com/privkit/privkit/issues).
- If not, post your proposal on the [issue tracker](https://github.com/privkit/privkit/issues) and provide as much detail as possible and source code (if any).

### To report an issue

- Check if there is any issue related to your problem already created on the [issue tracker](https://github.com/privkit/privkit/issues).
- If not, create an issue and provide as much detail as possible.
- If you have any source code that solves the issue, please add it to the issue and create a pull request with a descriptive title (check the _Commit Message Guidelines_).

## Commit Message Guidelines

It is important to write descriptive messages in your commits. By convention, the message should include one of the following prefixes:

- ENH: Enhancement, new feature
- BUG: Bug fix
- DOC: Additions/updates to documentation
- MAINT: Code maintenance
- TST: Additions/updates to tests
- BLD: Updates to the build process/scripts
- TYP: Type annotations
