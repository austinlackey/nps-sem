# NLP Survey Analysis Project

## Overview

Welcome to the NLP Survey Analysis project! This repository contains the code and documentation for a project that harnesses the power of Natural Language Processing (NLP) to unlock valuable insights from open-ended textual survey responses collected from visitors at various national parks.
To learn more about the project specifics, please read the following Medium article: [Parsing the Park-goer Perspective](https://medium.com/@austin-lackey/parsing-the-park-goer-perspective-0f432fd1b65f)

## Project Goals

The primary goal of this project is to leverage NLP techniques to extract patterns and information from textual responses, focusing on clustering responses by meaning. Unlike the National Park Service’s (NPS) quantitative analysis, this project aims to provide a methodological breakdown using popular tools like transformers, dimension reduction algorithms, clustering techniques, and visualization methods.

## Background

In 2023, the National Park Service released the SEM National Technical Report, offering comprehensive insights into park visitation for 2022. However, the NPS did not utilize the open-ended textual responses from the survey. This project aims to fill that gap by pairing the textual responses with modern
open-source Large Language Models (LLMs) to extract valuable insights.

## Methodology

The methodology involves using Hugging Face's transformer model for vectorization, followed by dimension reduction using algorithms like TSNE. Clustering algorithms, specifically k-means, are then applied to categorize similar responses. The project ensures reproducibility and consistency through code structure, model initialization, and tokenization processes.

## Code Reference and Reproducibility

The Jupyter notebook and Python code can be referenced in the [`clusterTransformers.py`](/clusterTransformers.py) document within this repository. Seeds are reset before major model operations to maintain reproducibility. All open-source packages used for this project are listed at the top of the code document.

