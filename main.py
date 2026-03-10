from kaggle_environments import make

def main():

    env = make("chess", debug=True)

    result = env.run(["chess_bot.py", "random"])
    print("Agent exit status/reward/time left: ")
    # look at the generated replay.json and print out the agent info
    for agent in result[-1]:
        print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
    print("\n")
    # render the game to an HTML file
    html = env.render(mode="html")
    with open("game.html", "w") as f:
        f.write(html)

    print("Game saved to game.html — open it in your browser.")

if __name__ == "__main__":
    main()