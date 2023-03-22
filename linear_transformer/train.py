"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

# data.py
from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

k_list = [512]

for k in k_list:

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def initialize_weights(m):
        if hasattr(m, "weight") and m.weight.dim() > 1:
            nn.init.kaiming_uniform(m.weight.data)

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
        k=k,
    ).to(device)

    # print('===============')
    # print(model)

    print(f"The model has {count_parameters(model):,} trainable parameters")
    model.apply(initialize_weights)
    optimizer = Adam(
        params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, verbose=True, factor=factor, patience=patience
    )

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        # print('%%%%%%%%%%')
        # print(iterator)
        for i, batch in enumerate(iterator):
            # print('&&&&&&&&&')
            # print(i)
            # print(batch)
            src = batch.src
            trg = batch.trg
            # print(src.shape)
            # print(src)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            print(
                "step :", round((i / len(iterator)) * 100, 2), "% , loss :", loss.item()
            )

        return epoch_loss / len(iterator)

    def evaluate(model, iterator, criterion):
        model.eval()
        epoch_loss = 0
        batch_bleu = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = criterion(output_reshape, trg)
                epoch_loss += loss.item()

                total_bleu = []
                for j in range(batch_size):
                    try:
                        # print('####yes####')
                        trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                        output_words = output[j].max(dim=1)[1]
                        output_words = idx_to_word(output_words, loader.target.vocab)
                        bleu = get_bleu(
                            hypotheses=output_words.split(), reference=trg_words.split()
                        )
                        total_bleu.append(bleu)
                    except:
                        # print('####NO####')
                        pass

                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        return epoch_loss / len(iterator), batch_bleu

    def run(total_epoch, best_loss):
        train_losses, test_losses, bleus = [], [], []
        for step in range(total_epoch):
            # print('$$$$$$')
            # print(total_epoch)
            # print(step)
            start_time = time.time()
            train_loss = train(model, train_iter, optimizer, criterion, clip)
            valid_loss, bleu = evaluate(model, valid_iter, criterion)
            end_time = time.time()

            if step > warmup:
                scheduler.step(valid_loss)

            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            bleus.append(bleu)
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_loss:
                best_loss = valid_loss

            # train_type = "original"
            # train_type = "norm_first"
            train_type = "linear"

            version = str((k, train_type))
            # if step >= 999:
            #     torch.save(
            #         model.state_dict(), "saved/model-" + k_str + "-{0}.pt".format(step)
            #     )

            f = open("../total_results/results/train_loss-" + version + ".txt", "w")
            f.write(str(train_losses))
            f.close()

            f = open("../total_results/results/bleu-" + version + ".txt", "w")
            f.write(str(bleus))
            f.close()

            f = open("../total_results/results/test_loss-" + version + ".txt", "w")
            f.write(str(test_losses))
            f.close()

            print(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
            # print(
            #     f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            # )
            # print(
            #     f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}"
            # )
            # print(f"\tBLEU Score: {bleu:.3f}")

    if __name__ == "__main__":
        run(total_epoch=epoch, best_loss=inf)