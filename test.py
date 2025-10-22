import math

def sequence(prev, T, cummulative):
    y = math.exp(prev)
    return (y/T)/(cummulative/T), T+1, cummulative + y

def gauss(x, mu, sigma):
    coeff = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * math.exp(exponent)

def normalize(x):
    total = max(x)
    return [i / total for i in x]

def sample(weights):
    probs = normalize(weights)
    r = 0.5
    cumulative = 0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1

val = 0
c = 1

with open("coeff.txt", "r") as f:
    num = float(f.read())
    if num <= 0:
        num = 42

n = sample([gauss(i, 100, max(int(num), 10)) for i in range(max(int(num), 10))]) * 2
if n < 10:
    n  = int(math.exp(n+5))
if n > 100:
    n = n % 100

for i in range(1, int(num)):
    val, _, c = sequence(val, i, c)
    # print(f"{i}: {val}")
rand = (n + val / max(1e-10,val**2))**2
if rand > 100:
    rand = rand%100
if rand < 10:
    rand = (rand * 694) % 100 
with open("coeff.txt", "w") as f:
    f.write(str(rand))
print(rand)