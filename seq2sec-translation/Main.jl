using Flux
using Flux: zeros, cat, unsqueeze, relu
import Flux: params
using Flux.Tracker
using Unicode
using Printf
using Random
using Knet: nll
using Plots

include("./Lang.jl")

ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
all_letters = ascii_lowercase * ascii_uppercase * " .,;'"
n_letters = length(all_letters)

const MAX_LENGTH = 10
const SOS_token = 1
const EOS_token = 2

function topk(tensor; rev=true)
  tensor |>
  enumerate |>
  collect |>
  s -> sort(s, by=(v -> v[2]), rev=rev)
end

function unicodeToAscii(s)
  s = [c for c in Unicode.normalize(s, :NFD) if Base.Unicode.category_abbrev(c) != "Mn"]
  join(s, "")
end

function normalizeString(s)
  lowercase(s) |>
  strip |>
  unicodeToAscii |>
  s -> replace(s, r"([.!?])" => s" \1") |>
  s -> replace(s, r"[^a-zA-Z.!?]+" => s" ")
end

function readLangs(lang1, lang2; rev=false)
  println("Reading lines...")

  lines = open("data/$lang1-$lang2.txt") do fp
    readlines(fp)
  end

  lines = map(l -> strip(l), lines)
  pairs = [[normalizeString(s) for s in split(l, "\t")] for l in lines]

  input_lang = Lang(lang1)
  output_lang = Lang(lang2)
  if rev == true
    pairs = [reverse(p) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  end

  input_lang, output_lang, pairs
end

eng_prefixes = [
  "i am ", "i m ",
  "he is", "he s ",
  "she is", "she s",
  "you are", "you re ",
  "we are", "we re ",
  "they are", "they re "
]

function filterPair(pairs)
  length(split(pairs[1], " ")) < 10 &&
  length(split(pairs[2], " ")) < 10 &&
  any(s -> startswith(pairs[2], s), eng_prefixes)
end

function filterPairs(pairs)
  [pair for pair in pairs if filterPair(pair)]
end

function prepareData(lang1, lang2; reverse=false)
  input_lang, output_lang, pairs = readLangs(lang1, lang2, rev=reverse)

  println("Read $(length(pairs)) sentence pairs")
  pairs = filterPairs(pairs)

  println("Trimmed to $(length(pairs)) sentence pairs")
  println("Counting words...")

  foreach(pairs) do pair
    addSentence(input_lang, pair[1])
    addSentence(output_lang, pair[2])
  end

  println("Counted words:")
  println(input_lang.name, " ", input_lang.n_words)
  println(output_lang.name, " ", output_lang.n_words)

  input_lang, output_lang, pairs
end

input_lang, output_lang, ps = prepareData("eng", "fra", reverse=true)

mutable struct Embedded
  w

  function Embedded(input_size, hidden_size)
    inst = new()
    inst.w = param(randn(input_size, hidden_size))
    inst
  end
end

(m::Embedded)(input::AbstractArray) = m.w[input,:]

Flux.@treelike Embedded

mutable struct EncoderRNN
  hidden_size

  embedded
  gru

  function EncoderRNN(input_size, hidden_size)
    inst = new()
    inst.hidden_size = hidden_size
    inst.embedded = Embedded(input_size, hidden_size)
    inst.gru = Flux.GRUCell(hidden_size, hidden_size)
    inst
  end
end

function forward(m::EncoderRNN, input, hidden) 
  x = m.embedded(input)
  out = reshape(x, 1, 1, :)
  out, hidden = m.gru(out[:], hidden[:])
  out, hidden
end

Flux.@treelike EncoderRNN

function initHidden(m::EncoderRNN)
  Flux.zeros(1, 1, m.hidden_size)
end

mutable struct DecoderRNN
  hidden_size

  embedded
  gru
  out
  softmax

  function DecoderRNN(hidden_size, output_size) 
    inst = new()
    inst.hidden_size = hidden_size
    inst.embedded = Embedded(hidden_size, output_size)
    inst.gru = Flux.GRUCell(hidden_size, hidden_size)
    inst.out = Flux.GRUCell(hidden_size + outputsize, output_size)
    inst.softmax = s -> logsoftmax(s, dims=1)
    inst
  end
end

function forward(m::DecoderRNN, input, hidden)
  out = m.embedding(input) |> Flux.relu
  out, hide = m.gru(out[:], hide[:])
  out = m.softmax(rnn.out(out[1, :, :]))
  out, hide
end

function initHidden(rnn::DecoderRNN)
  Flux.zeros(1, 1, rnn.hidden_size)
end

mutable struct AttnDecoderRNN
  hidden_size
  output_size
  dropout_p
  max_length

  embedded
  attn
  attn_combine
  dropout
  gru
  out

  function AttnDecoderRNN(hidden_size, output_size; dropout_p=0.1, max_length=MAX_LENGTH)
    inst = new()
    inst.hidden_size = hidden_size
    inst.output_size = output_size
    inst.dropout_p = dropout_p
    inst.max_length = max_length

    inst.embedded = Embedded(output_size, hidden_size)
    inst.attn = Dense(hidden_size * 2, max_length) # => f(x) = w * x + b
    inst.attn_combine = Dense(hidden_size * 2, hidden_size)
    inst.dropout = Flux.Dropout(dropout_p)
    inst.gru = Flux.GRUCell(hidden_size, hidden_size)
    inst.out = Dense(hidden_size, output_size)
    inst
  end
end

function forward(m::AttnDecoderRNN, input, hidden, encoder_outputs)
  embedded = m.embedded(input) |> m.dropout
  hidden = reshape(hidden, 1, 1, :)

  attn_weigths = cat(embedded[:], hidden[:], dims=2)
  attn_weigths = m.attn(attn_weigths[:])
  attn_weigths = Flux.softmax(attn_weigths)
  attn_weigths = reshape(attn_weigths, 1, :)

  attn_applied = sum(encoder_outputs.*transpose(attn_weigths), dims=1)
  attn_applied = reshape(attn_applied, 1, 1, :)

  out = cat(embedded[1,:,:], attn_applied[1,:,:], dims=2)

  out = m.attn_combine(out[:])
  out = unsqueeze(out, 1)
  out = relu(out[:])
  out, hide = m.gru(out[:], hidden[:])

  out = reshape(out, 1, 1, :)
  out = Flux.logsoftmax(m.out(out[:]))

  out, hide, attn_weigths
end

function initHidden(rnn::AttnDecoderRNN)
  Flux.zeros(1, 1, rnn.decoder_hidden_size)
end

Flux.@treelike AttnDecoderRNN

function indexesFromSentence(lang, sentence)
  [lang.word2index[word] for word in split(sentence, " ")]
end

function tensorFromSentence(lang, sentence)
  indexes = indexesFromSentence(lang, sentence)
  push!(indexes, EOS_token)
  indexes |> v -> view(v, :, 1:1)
end

function tensorsFromPair(pair)
  input_tensor = tensorFromSentence(input_lang, pair[1])
  target_tensor = tensorFromSentence(output_lang, pair[2])
  input_tensor, target_tensor
end

teacher_forcing_ratio = 0.5

function train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion;
    max_length=MAX_LENGTH,
    learning_rate=0.01
 )
  encoder_hidden = initHidden(encoder)

  Flux.reset!(encoder)
  Flux.reset!(decoder)

  input_length = size(input_tensor)[1]
  target_length = size(target_tensor)[1]
  encoder_outputs = zeros(max_length, encoder.hidden_size)

  loss = 0

  for ei=1:input_length
    encoder_output, encoder_hidden = forward(encoder, input_tensor[ei,:], encoder_hidden)
    encoder_outputs[ei,:,:] = permutedims(reshape(encoder_output.data, 1, :), (2, 1))
    # encoder_outputs[ei,:] = encoder_output.data
  end

  decoder_input = reshape([SOS_token], 1, :)
  decoder_hidden = encoder_hidden

  if rand() < teacher_forcing_ratio
    use_teacher_forcing = true
  else
    use_teacher_forcing = false
  end

  if use_teacher_forcing
    for di=1:target_length
      decoder_output, decoder_hidden, decoder_attention = forward(
        decoder,
        decoder_input,
        decoder_hidden,
        encoder_outputs
      )

      loss += criterion(decoder_output, target_tensor[di,:,:], dims=1)

      decoder_input = target_tensor[di,:,:]
    end
  else
    for di=1:target_length
      decoder_output, decoder_hidden, decoder_attention = forward(
        decoder,
        decoder_input,
        decoder_hidden,
        encoder_outputs
      )

      top = topk(decoder_output)[1:1]
      topi = map(v -> v[1], top)
      topv = map(v -> v[2], top)

      decoder_input = reshape(topi[:], 1, :)
      loss += criterion(decoder_output, target_tensor[di,:,:], dims=1)

      if decoder_input[1] == EOS_token
        break
      end
    end
  end

  back!(loss)

  encoder_optimizer()
  decoder_optimizer()

  return loss.data / target_length
end

plot_losses = []

function trainIters(encoder, decoder, n_iters; print_every=1000, plot_every=100, learning_rate=0.01)
  print_loss_total = 0
  plot_loss_total = 0

  encoder_optimizer = Flux.SGD(params(encoder), learning_rate)
  decoder_optimizer = Flux.SGD(params(decoder), learning_rate)

  training_pairs = [tensorsFromPair(ps[rand(1:length(ps))]) for i in 1:n_iters]
  criterion = nll

  for iter in 2:(n_iters+1)
    training_pair = training_pairs[iter-1]
    input_tensor = training_pair[1]
    target_tensor = training_pair[2]

    loss = train(
      input_tensor,
      target_tensor,
      encoder,
      decoder,
      encoder_optimizer,
      decoder_optimizer,
      criterion,
      learning_rate=learning_rate
    )
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0

      @printf "%s (%d %d%%) %.4f\n" "" iter ((iter/n_iters)*100) print_loss_avg
    end

    if iter % plot_every == 0
      plot_loss_avg = plot_loss_total / plot_every
      push!(plot_losses, plot_loss_avg)
      plot_loss_total = 0
    end
  end

  plot(plot_losses)
end


# function bmm(A, B)
#   bs,m,n = size(A)
#   bs,n,k = size(B)
#   bmm(A, B, Flux.similar(A, (bs, m, k)))
# end
# 
# function bmm(A, B, C)
#   bsa,ma,ka = size(A)
#   bsb,kb,nb = size(B)
# 
#   C = TrackedArray(C)
# 
#   for i=1:bsa
#     a = Flux.view(A.data, i, :, :)
#     b = Flux.view(B.data, i, :, :)
#     C.data[i,:,:] = a * b
#   end
# 
#   C
# end

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

trainIters(encoder1, attn_decoder1, 10000, print_every=100)
# trainIters(encoder1, attn_decoder1, 1000, print_every=10, plot_every=10)


function evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH)
  input_tensor = tensorFromSentence(input_lang, sentence)
  input_length = size(input_tensor)[1]
  encoder_hidden = initHidden(encoder)
  encoder_outputs = zeros(max_length, encoder.hidden_size)

  for ei=1:input_length
    encoder_output, encoder_hidden = forward(
      encoder, 
      [input_tensor[ei]], 
      encoder_hidden
    )
    encoder_outputs[ei,:] .+= encoder_output.data
  end

  decoder_input = reshape([SOS_token], 1, :)
  decoder_hidden = encoder_hidden

  decoded_words = []
  decoder_attentions = zeros(max_length, max_length)

  gdi = 1
  for di=1:max_length
    gdi = di

    decoder_output, decoder_hidden, decoder_attention = forward(
      decoder,
      decoder_input,
      decoder_hidden,
      encoder_outputs
    )
    decoder_attentions[di,:] = decoder_attention.data

    top = topk(decoder_output, rev=true)[1:1]
    topi = map(v -> v[1], top)
    topv = map(v -> v[2], top)

    if topi[1] == EOS_token
      push!(decoded_words, "<EOS>")
      break
    else
      push!(decoded_words, output_lang.index2word[topi[1]])
    end

    decoder_input = reshape(topi[:], 1, :)
  end

  decoded_words, decoder_attentions[1:gdi]
end

function evaluateRandomly(encoder, decoder, n=10)
  for i=1:n
    pair = ps[rand(1:length(ps))]

    println("> ", pair[1])
    println("= ", pair[2])

    output_words, attentions = evaluate(encoder, decoder, pair[1])
    output_sentence = join(output_words, " ")

    println("< ", output_sentence)
    println("")
  end
end

evaluateRandomly(encoder1, attn_decoder1)

words = [
  "elle a cinq ans de moins que moi .",
  "elle est trop petit .",
  "je ne crains pas de mourir .",
  "c est un jeune directeur plein de talent .",
]

for i=1:length(words)
  word = words[i]
  output_words, attenstions = evaluate(encoder1, attn_decoder1, word)

  println("> ", word)
  println("< ", join(output_words, " "))
  println("")
end
