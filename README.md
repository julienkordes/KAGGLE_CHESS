# Chess PPO Bot

A reinforcement learning agent trained to play chess using **Proximal Policy Optimization (PPO)**, built with the [BBRL](https://github.com/osigaud/bbrl) library.

## Overview

This project implements a PPO-based agent that learns to play chess from scratch through self-play and environment interaction. The agent observes the board state, selects legal moves, and is rewarded based on game outcomes.

## Features

- PPO algorithm with clipped surrogate objective
- Chess environment wrapper compatible with BBRL's agent abstraction
- Legal move masking to prevent invalid actions
- Self-play training loop
- Evaluation against random and greedy opponents
