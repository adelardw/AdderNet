import cpuinfo
from time import perf_counter
from loaders import *
from lightning_module import *
from pipeline.cfg_loader import load_cfg
import GPUtil


_, _, test_loader = get_loaders(speech_dirs=["dev-clean.tar", "test-clean.tar"],
                                                    noise_dir="./wham_noise//wham_noise",
                                                    batch_size=1,
                                                    padding_strategy=None)

cfg_path ='configs/denoise_model_v1_cfg.yaml'
ckpt_path = 'ep_130.ckpt'
model_cfg = load_cfg(cfg_path)

metrics = dict(snratio = SNR(),
                sdratio = SDR(),
                sisdratio = SISDR(),
                sisnratio = SISNR())

model = SpectrogramLightningModelUnet.load_from_checkpoint(ckpt_path,
                                                            **model_cfg)
info = cpuinfo.get_cpu_info()
gpus = GPUtil.getGPUs()
print(" "*50)
print("-"*20 + 'DEVICE INFO' + "-"*20)
print(f"Процессор: {info['brand_raw']}")
print(f"Количество ядер: {info['count']}")
for g in gpus:
    print(f"Видеокарта: {g.name}")
    print(f"Память: {g.memoryTotal} MB")
    print(f"Используется памяти: {g.memoryUsed} MB")
    print(f"Загрузка GPU: {g.load * 100}%")
print("-"*20 + '----------' + "-"*20)   
    
def compute_metrics(cleaned_wf, clean_wf, metric):
        cleaned_wf = cleaned_wf.detach().cpu()
        clean_wf = clean_wf.detach().cpu()
        cleaned_wf_shape = cleaned_wf.shape[-1]
        clean_wf_shape = clean_wf.shape[-1]
        if cleaned_wf.shape[1] != 1:
            cleaned_wf = cleaned_wf.sum(1, keepdims=True)
        if clean_wf.shape[1] != 1:
            clean_wf = clean_wf.sum(1, keepdims=True)
        if clean_wf_shape == min(clean_wf_shape, cleaned_wf_shape):
            cleaned_wf = cleaned_wf[:, :, :clean_wf_shape]
        else:
            clean_wf = clean_wf[:, :, :cleaned_wf_shape]

        metric['snratio'].update(cleaned_wf, clean_wf)
        metric['sisdratio'].update(cleaned_wf, clean_wf)
        metric['sdratio'].update(cleaned_wf, clean_wf)
        metric['sisnratio'].update(cleaned_wf, clean_wf)
        
def bench_model(model,metrics, num_iters=100, loader=test_loader):
    model.eval()
    cpu = []
    gpu = []
    for i, batch in enumerate(loader):
        
        model = model.to('cpu')
        mixed, clean = batch
        
        mixed = mixed.to('cpu')
        
        start = perf_counter()
        out = model.run(mixed)
        delta =perf_counter() - start
        cpu.append(delta)
        
        model = model.to('cuda')
        mixed = mixed.to('cuda')
        clean = clean.to('cuda')
        
        
        start = perf_counter()
        out = model.run(mixed)
        delta = perf_counter() - start
        gpu.append(delta)

        out = torch.tensor(out).to('cuda')
        compute_metrics(out, mixed, metrics)
        
        if (i + 1) % (num_iters) == 0:
            break
        
    snratio = metrics['snratio'].compute()
    sisdratio = metrics['sisdratio'].compute()
    sdratio = metrics['sdratio'].compute()
    sisnratio = metrics['sisnratio'].compute()
    
    metric_values = dict(snr = snratio.item(),
                        sisdr = sisdratio.item(),
                        sdr = sdratio.item(),
                        sisnr = sisnratio.item())
    
    return cpu, gpu, metric_values



cpu, gpu,metric_values = bench_model(model, metrics)



print('-'*20 + '+CPU+' + '-'*20)
print('Avg CPU Inference [s] : ', torch.tensor(cpu).mean().item())
print('-'*20 + '-----' + '-'*20)
print('-'*20 + '+GPU+' + '-'*20)
print('Avg GPU Inference [s] : ', torch.tensor(gpu).mean().item())
print('-'*20 + '-----' + '-'*20)
print('-'*19 + 'METRICS' + '-'*19 )
print(f"SNR [dB]: {metric_values['snr']}"),
print(f"SDR [dB]: {metric_values['sdr']}")
print(f"SI-SDR [dB]: {metric_values['sisdr']}")
print(f"SI-SNR [dB]: {metric_values['sisnr']}")
print('-'*19 + '-------' + '-'*19 )
print(" "*50)