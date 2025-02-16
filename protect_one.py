import argparse
from torch.utils.data import DataLoader
import soundfile as sf

import vits.utils as utils
from vits.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from toolbox import build_models
from protect import minimize_error

def main():
    parser = argparse.ArgumentParser(description="Protect single audio (no fakes) by using real dataset info.")
    parser.add_argument("--filelist", type=str, required=True,
                        help="A .txt file that contains exactly one line: /path/to/audio.wav|<speaker_id>|<text>")
    parser.add_argument("--output_path", type=str, default="protected.wav",
                        help="Where to save the perturbed audio.")
    args = parser.parse_args()

    device = 'cuda'
    hps = utils.get_hparams_from_file("configs/onespeaker_vits.json")
    net_g, net_d = build_models(hps, checkpoint_path="checkpoints/pretrained_ljs.pth")
    net_g, net_d = net_g.to(device), net_d.to(device)
    net_g.eval()
    net_d.eval()


    dataset = TextAudioSpeakerLoader(args.filelist, hps.data)
    collate_fn = TextAudioSpeakerCollate()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    epsilon = 8/255.0
    alpha = epsilon / 10.0
    max_epoch = 200
    model_name = "VITS"


    for batch_data in data_loader:
        noise, final_loss = minimize_error(
            hps=hps,
            nets=[net_g, net_d],
            epsilon=epsilon,
            alpha=alpha,
            max_epoch=max_epoch,
            batch_data=batch_data,
            mode="POP",
            model_name=model_name
        )

        wav_tensor = batch_data[4]  # [1, 1, T]
        sr = hps.data.sampling_rate

        protected_wav = (wav_tensor.cpu() + noise.cpu()).clamp(-1., 1.).detach().cpu().numpy().squeeze()
        sf.write(args.output_path, protected_wav, sr)
        print(f"[INFO] Saved protected audio to {args.output_path}")
        break

if __name__ == "__main__":
    main()
