import numpy as np
import easygui

start = ['Rock', 'Paper', 'Scissors']

def inputAct():
    cur_act = input("Enter Activity(Rock,Paper,Scissors): ")
    if cur_act != 'Rock' and cur_act != 'Paper' and cur_act != 'Scissors':
        print('The choice was made for you')
        cur_act = np.random.choice(start, replace=True, p=p_start)
    return cur_act

def printPredictionResult(res):
    if prediction != res:
        print('prediction was wrong')
    else:
        print('prediction was correct')

def checkGamePart():
    if current_state == "Scissors" and activity == "Scissors":
        printPredictionResult('tie')
        print("result: tie")
    elif current_state == "Rock" and activity == "Scissors":
        printPredictionResult('win')
        print("result: win")
    elif current_state == "Paper" and activity == "Scissors":
        printPredictionResult('lose')
        print("result: lose")
    elif current_state == "Scissors" and activity == "Rock":
        printPredictionResult('lose')
        print("result: lose")
    elif current_state == "Rock" and activity == "Rock":
        printPredictionResult('tie')
        print("result: tie")
    elif current_state == "Paper" and activity == "Rock":
        printPredictionResult('win')
        print("result: win")
    elif current_state == "Scissors" and activity == "Paper":
        printPredictionResult('win')
        print("result: win")
    elif current_state == "Rock" and activity == "Paper":
        printPredictionResult('lose')
        print("result: lose")
    elif current_state == "Paper" and activity == "Paper":
        printPredictionResult('tie')
        print("result: tie")



p_start = [0.3, 0.43, 0.27]

t = ['Rock', 'Paper', 'Scissors']
tm = {
     'Rock': [0.00, 0.11, 0.89],
     'Paper': [0.17, 0.33, 0.50],
     'Scissors': [0.60, 0.20, 0.20]
}

e = ['0', '1', '-1']
ep = [
    [0.22, 0.33, 0.45],
    [0.50, 0.17, 0.33],
    [0.47, 0.33, 0.20]
]

initial_state = np.random.choice(start, replace=True, p=p_start)
print(initial_state)



for game in range(1, 31):
    print('Game {game}'.format(game=game))
    current_state = np.random.choice(start, replace=True, p=p_start)
    if initial_state == 'Rock':
        if current_state == 'Rock':
            prob = tm['Rock'][0]
        elif current_state == 'Paper':
            prob = tm['Rock'][1]
        elif current_state == 'Scissors':
            prob = tm['Rock'][2]
    elif initial_state == 'Paper':
        if current_state == 'Rock':
            prob = tm['Paper'][0]
        elif current_state == 'Paper':
            prob = tm['Paper'][1]
        elif current_state == 'Scissors':
            prob = tm['Paper'][2]
    elif initial_state == "Scissors":
        if current_state == 'Rock':
            prob = tm['Scissors'][0]
        elif current_state == 'Paper':
            prob = tm['Scissors'][1]
        elif current_state == 'Scissors':
            prob = tm['Scissors'][2]

    initial_state = current_state

    if prob <= 0.33:
        prediction = 'lose'
        print('pred: '+prediction)
    elif 0.33 < prob <= 0.68:
        prediction = 'tie'
        print('pred: '+prediction)
    elif prob > 0.68:
        prediction = 'win'
        print('pred: '+prediction)

    activity = inputAct()
    print(activity)
    checkGamePart()












