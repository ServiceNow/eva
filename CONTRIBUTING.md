# Contributing to EVA

Thank you for your interest in contributing to EVA!

This document should be able to guide contributors in their different types of contributions.

Just want to ask a question? Open a topic on our [Discussion page](https://github.com/ServiceNow/eva/discussions).

## Before You Start

We welcome contributions, but want to be upfront: our team has limited bandwidth for reviewing pull requests, and not every contribution will align with the direction we're taking EVA. To make the best use of everyone's time:

- **For bug fixes and small improvements**: Feel free to open a PR directly.
- **For larger features or significant changes**: Please [open an issue](https://github.com/ServiceNow/eva/issues/new/choose) or [start a discussion](https://github.com/ServiceNow/eva/discussions) **before** writing any code. This lets us confirm the change fits our vision and saves you from investing time in work we may not be able to merge.

Check out our [roadmap and current limitations](https://servicenow.github.io/eva/#limitations) to see the areas we're most interested in improving. Contributions that address items on the roadmap are much more likely to be reviewed and merged quickly.

## Get Your Environment Setup

Go to our [Quick Start](README.md) section in the README to get set up.

## How to Submit a Bug Report

[Open an issue on GitHub](https://github.com/ServiceNow/eva/issues/new/choose) and select "Bug report". If you are not sure whether it is a bug or not, submit an issue and we will be able to help you.

Issues with reproducible examples are easier to work with. Do not hesitate to provide your configuration with generated data if need be.

If you are familiar with the codebase, providing a unit test is helpful, but not mandatory.

## How to Submit Changes

First, open an issue describing your desired changes, if it does not exist already.
1. [Fork the repo to your own account](https://github.com/ServiceNow/eva/fork).
2. Clone your fork of the repo locally.
3. Make your changes (the fun part).
4. Commit and push your changes to your fork.
5. [Open a pull request](https://github.com/ServiceNow/eva/compare) with your branch.
6. Once a team member approves your changes, we will merge the pull request promptly.

### Guidelines for a Good Pull Request

When coding, pay special attention to the following:

- Your code should be well commented for non-trivial sections, so it can be easily understood and maintained by others, but not over-commented. Good variable names and functions are your best friend.
- Do not expose any personal or sensitive data.
- Add unit tests when a notable functionality has been added or changed.
