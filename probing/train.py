import torch
from open_lm.file_utils import pt_load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def load_model(args, model):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
    return start_epoch


def load_optimizer(args, model, optimizer):
    potential_checkpoint = args.resume.replace("epoch_", "optimizer_")
    if check_exists(potential_checkpoint):
        checkpoint = pt_load(potential_checkpoint, map_location="cpu")
    else:
        checkpoint = pt_load(args.resume, map_location="cpu")
    if "optimizer" in checkpoint:
        if optimizer is not None:
            osd = checkpoint["optimizer"]
            if args.fsdp:
                osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
            optimizer.load_state_dict(osd)


args.vocab_size = 50432


def train(model, optimizer, data, args):
    loss = torch.nn.CrossEntropyLoss()
    autocast = get_autocast(args.precision)
    with autocast():
        forward_start = time.time()
        inputs, targets = sample_chunk(texts, args)
        out, _, _ = model(inputs)
        print(time.time() - forward_start)
        logit_m.update(torch.mean(out).item())
        total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))

    backward_start = time.time()
    total_loss.backward()
    print(time.time() - backward_start)

    args.grad_clip_norm = 1
    if args.grad_clip_norm is not None:
        if isinstance(model, FSDP):
            model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm, norm_type=2.0
            )
    optimizer.step()


def evaluate(model, data, start_epoch, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(
        start_epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader

    loss = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(dataloader):
        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)

        data_time_m.update(time.time() - end)

        with autocast():
            inputs = texts[:, : args.seq_len - 1]
            targets = texts[:, 1 : args.seq_len]
            out, _ = model(inputs)
            total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))

    if is_master(args):
        print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
    return log_data
