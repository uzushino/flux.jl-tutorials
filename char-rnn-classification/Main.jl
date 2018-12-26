using Flux: param, zeros, cat, glorot_uniform
using Flux.Tracker
using Plots
using Glob
using Unicode
using Printf 
using Random
using Knet: nll, logsoftmax

findFiles(path) = Glob.glob(path)

ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
all_letters = ascii_lowercase * ascii_uppercase * " .,;'"
n_letters = length(all_letters)

function unicodeToAscii(s)
  Unicode.normalize(s, :NFD) |> 
  s -> filter(c -> in(c, all_letters), s) |> 
  s -> join(s, "")
end

category_lines = Dict()
all_categories = []

function readLines(filename)
  lines = open(filename, "r") do fp
    readlines(fp)
  end

  map(l -> unicodeToAscii(l), lines)
end

for filename in findFiles("data/names/*.txt")
  category = (filename |> basename |> splitext)[1]
  push!(all_categories, category)
  lines = readLines(filename)
  category_lines[category] = lines
end

n_categories = length(all_categories)

function letterToIndex(letter)
  findfirst(c -> c == letter, split(all_letters, ""))
end

function letterToTensor(letter)
  tensor = zeros(1, n_letters)
  tensor[1, letterToIndex(letter)] = 1
  tensor
end

function lineToTensor(line)
  tensor = zeros(length(line), 1, n_letters)

  for (li, letter) in enumerate(line)
    tensor[li, 1, string(letter) |> letterToIndex] = 1
  end

  tensor
end

struct RNN
  W1
  b1
  W2
  b2

  hidden_size
  i2h
  i2o
  softmax

  RNN(input_size, hidden_size, output_size) = begin
    W1 = param(glorot_uniform(hidden_size, input_size + hidden_size))
    b1 = param(glorot_uniform(hidden_size))
    i2h(x) = W1 * x + b1

    W2 = param(glorot_uniform(output_size, input_size + hidden_size))
    b2 = param(glorot_uniform(output_size))
    i2o(x) = W2 * x + b2
    softmax(x) = logsoftmax(x, dims=1)

    new(W1, b1, W2, b2, hidden_size, i2h, i2o, softmax)
  end
end

function initHidden(rnn::RNN)
  zeros(1, rnn.hidden_size)
end

function forward(rnn::RNN, input, hide)
  combined = cat(input, hide, dims=2)

  hide = rnn.i2h(combined[:])
  out = rnn.i2o(combined[:]) |> rnn.softmax

  out, hide
end

n_hidden = 128
input_size = n_letters
hidden_size = n_hidden
output_size = n_categories

rnn = RNN(input_size, hidden_size, output_size)

input = lineToTensor("Albert")
output, next_hidden = forward(rnn, input[1,:,:], zeros(1, n_hidden))

function topk(tensor; rev=true)
  tensor |> 
  enumerate |> 
  collect |> 
  s -> sort(s, by=(v -> v[2]), rev=rev)
end

function categoryFromOutput(guess)
  i, n = topk(guess, rev=true)[1]
  category_i = i
  all_categories[i], category_i
end

randomChoice(l) = l[rand(1:length(l))]

function randomTrainingExample()
  category = randomChoice(all_categories)
  line = randomChoice(category_lines[category])

  index = findfirst(c -> c == category, all_categories)
  category_tensor = reshape([index], 1, :)
  line_tensor = lineToTensor(line)

  category, line, category_tensor, line_tensor
end

for i=1:10
  category, line, category_tensor, line_tensor = randomTrainingExample()
  println("category = $category / line = $line")
end

criterion = nll
learning_rate = 0.005

current_loss = 0
n_iters = 100_000
print_every = 5_000
plot_every = 1_000
all_losses = []

for i=1:n_iters
  rnn.W1.grad .= 0.0
  rnn.W2.grad .= 0.0
  rnn.b1.grad .= 0.0
  rnn.b2.grad .= 0.0

  category, line, category_tensor, line_tensor = randomTrainingExample()

  hidden = param(zeros(1, hidden_size))
  output = 1
  
  s = size(line_tensor)[1]
  for j=1:s
    output, hidden = forward(rnn, line_tensor[j,:,:], hidden)
    hidden = reshape(hidden, 1, :)
  end

  loss = criterion(output, category_tensor, dims=1)
  @show loss
  back!(loss)
  @show rnn.W1.grad
  a += 1
  global current_loss += loss

  rnn.W1.data .-= learning_rate .* rnn.W1.grad
  rnn.W2.data .-= learning_rate .* rnn.W2.grad
  rnn.b1.data .-= learning_rate .* rnn.b1.grad
  rnn.b2.data .-= learning_rate .* rnn.b2.grad
  
  if i % print_every == 0
    guess, guess_i = categoryFromOutput(output)

    if guess == category 
      correct = "✓" 
    else 
      correct = "✗ ($category)"
    end

    @printf("%d %d%% %.4f %s / %s %s\n", i, i / n_iters * 100, loss, line, guess, correct)
  end

  if i % plot_every == 0
    push!(all_losses, (current_loss / plot_every))
    current_loss = 0
  end
end

# using Plots
# plot(map(v -> v.data, all_losses))

confusion = zeros(n_categories, n_categories)
n_confusion = 10_000

function evaluate(line_tensor)
  hidden = zeros(1, hidden_size)
  out = 1

  s = size(line_tensor)[1]
  for j=1:s
    out, hidden = forward(rnn, line_tensor[j,:,:], hidden)
    hidden = reshape(hidden, 1, :)
  end

  out
end

function predict(input_line; n_predictions=3)
  @printf "\n> %s\n" input_line
  out = evaluate(lineToTensor(input_line))

  top = topk(out, rev=true)[1:n_predictions]
  topi = map(v -> v[1], top)
  topv = map(v -> v[2], top)
  predictions = []

  for i=1:n_predictions
    value = topv[i].data
    category_index = topi[i]
    all_categories[category_index]

    @printf "(%.2f) %s\n" value all_categories[category_index]
    push!(predictions, [value, all_categories[category_index]])
  end
end

predict("Dovesky")

predict("Jackson")

predict("Satoshi")