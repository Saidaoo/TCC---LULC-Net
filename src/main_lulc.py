from skimage import io
import os, time
import torch
import numpy as np
import pandas as pd
from project_utils import load_loss_weights

from dataset import DatasetIcmbio
from trainer import Trainer
from models_lulc import build_lulc_model
from project_utils import clear, convert_to_color, make_optimizer, seed_everything, visualize_augmentations

import sys
import os

# Garante que estamos importando da pasta src/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"✅ Importando de: {current_dir}")
print(f"✅ Arquivos aqui: {os.listdir(current_dir)}")

# Agora faça os imports normais
from skimage import io
import time
import torch
import numpy as np
import pandas as pd
from project_utils import load_loss_weights, batch_mean_and_sd, clear, convert_to_color, make_optimizer, seed_everything
from dataset import DatasetIcmbio
from trainer import Trainer
from models_lulc import build_lulc_model

def is_save_epoch(epoch, ignore_epoch=0):
    return params['save_epoch'] is not None and epoch % params['save_epoch'] == 0 and epoch != ignore_epoch
    
def forward(self, x):
    print(f"🔍 DEBUG DIMENSÕES - Input: {x.shape}")
    
    # Encoder
    e1 = self.enc1(x)
    print(f"🔍 DEBUG DIMENSÕES - e1: {e1.shape}")
    
    e2 = self.enc2(e1)
    print(f"🔍 DEBUG DIMENSÕES - e2: {e2.shape}")
    
    e3 = self.enc3(e2)
    print(f"🔍 DEBUG DIMENSÕES - e3: {e3.shape}")
    
    e4 = self.enc4(e3)
    print(f"🔍 DEBUG DIMENSÕES - e4: {e4.shape}")
    
    # Bridge
    b = self.bridge(e4)
    print(f"🔍 DEBUG DIMENSÕES - bridge: {b.shape}")
    
    # Decoder
    d4 = self.dec4(b)
    print(f"🔍 DEBUG DIMENSÕES - d4: {d4.shape}, e4: {e4.shape}")
    d4 = d4 + e4  # Aqui está o problema!
    
    d3 = self.dec3(d4)
    d3 = d3 + e3
    
    d2 = self.dec2(d3) 
    d2 = d2 + e2
    
    d1 = self.dec1(d2)
    d1 = d1 + e1
    
    final = self.final(d1)
    return final

class LossFN:
    CROSS_ENTROPY = 'cross_entropy'
    FOCAL_LOSS = 'focal_loss'
    DICE = 'DICE'
    JACCARD = 'JACCARD'
    TVERSKY = 'TVERSKY'

class ModelChooser:
    SEGNET_MODIFICADA = 'segnet_modificada'
    UNET = 'unet'
    SEGFORMER = 'segformer'
    DEEPLABV3PLUS = 'deeplabv3plus'
    LULC_NET = 'lulc_net'

class Callback():

    def __init__(self, patience = 10, min_value = 66):
        self.PATIENCE = patience
        self.COUNTER = 0
        self.MIN_LIMIT = min_value
        self.BEST_VALUE = 0
        self.BEST_TRAINER = []

    def patience_loss(self, epoch):
        if trainer.epoch_loss[epoch-1] < self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_loss[epoch-1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_loss[epoch-1] >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Loss Too High | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False
    
    def patience_acc(self, epoch):
        if trainer.epoch_acc[epoch-1] > self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_acc[epoch-1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_acc[epoch-1] <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False
    
    def patience_acc_val(self, avg_acc):
        if avg_acc > self.MIN_LIMIT and avg_acc > self.BEST_VALUE:
            self.BEST_VALUE = avg_acc
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val acc < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_f1_val(self, f1):
        if f1 > self.MIN_LIMIT and f1 > self.BEST_VALUE:
            self.BEST_VALUE = f1
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val F1-Score < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_iou_val(self, iou):
        if iou > self.MIN_LIMIT and iou > self.BEST_VALUE:
            self.BEST_VALUE = iou
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val mIoU < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_loss_val(self, avg_loss):
        if avg_loss >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        elif avg_loss < self.BEST_VALUE:
            self.BEST_VALUE = avg_loss
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False


def weights_calculator_loss(params, train_labels):
    try:
        if params['loss']['name'] == LossFN.CROSS_ENTROPY:
            if params['loss']['params']['weights'] == 'equal':
                params['weights'] = torch.ones(params['n_classes'])
            elif params['loss']['params']['weights'] == 'calculate':
                if os.path.exists('./loss_weights.npy'):
                    loss_weights = load_loss_weights('./loss_weights.npy')
                    params['weights'] = torch.from_numpy(loss_weights['weights']).float()
                else:
                    import utils.weights_calculator as wc
                    loss_weights, _ = wc.WeightsCalculator(train_labels, params['classes'], dev=False).calculate_and_save()
                    params['weights'] = torch.from_numpy(loss_weights).float()
        elif params['loss']['name'] == LossFN.FOCAL_LOSS:
            params['weights'] = torch.ones(params['n_classes'])

        print("🎯 Pesos das classes para loss:")
        for i, (cls, weight) in enumerate(zip(params['classes'], params['weights'])):
            print(f"   {cls}: {weight:.4f}")
    except Exception as e:
        print(f"❌ Erro ao calcular pesos: {e}")
        raise e

def print_training_summary(params, train_loader, val_loader, test_loader):
    """Debug completo do setup de treinamento"""
    print("\n" + "="*80)
    print("📊 RESUMO DO TREINAMENTO")
    print("="*80)
    print(f"🏷️  Modelo: {params['model']['name']}")
    print(f"📈 Épocas: {params['maximum_epochs']}")
    print(f"📦 Batch Size: {params['bs']}")
    print(f"🖼️  Tamanho da Janela: {params['window_size']}")
    print(f"🎯 Loss: {params['loss']['name']}")
    print(f"⚡ Otimizador: {params['optimizer_params']['optimizer']}")
    print(f"📚 Learning Rate: {params['optimizer_params']['lr']}")
    print(f"🎲 Augmentation: {params['augment']}")
    print(f"💾 Cache: {params['cache']}")
    print(f"📁 Pasta de resultados: {params['results_folder']}")
    print(f"🎯 Classes: {params['n_classes']}")
    print(f"📊 Datasets:")
    print(f"   - Treino: {len(train_loader.dataset)} amostras")
    print(f"   - Validação: {len(val_loader.dataset)} amostras") 
    print(f"   - Teste: {len(test_loader.dataset)} amostras")
    print(f"   - Batches por época: {len(train_loader)}")
    print("="*80 + "\n")

def print_epoch_progress(epoch, acc_train, f1score_train, iou_train, acc_val, f1score_val, iou_val, start_epoch_time):
    """Debug do progresso por época"""
    epoch_time = time.time() - start_epoch_time
    print(f"\n⏰ Época {epoch} concluída em {epoch_time:.2f}s")
    print(f"📊 TREINO  | Acc: {acc_train:.2f}% | F1: {f1score_train:.4f} | IoU: {iou_train:.4f}")
    print(f"🎯 VALIDAÇÃO | Acc: {acc_val:.2f}% | F1: {f1score_val:.4f} | IoU: {iou_val:.4f}")
    
    # Calcula melhorias
    if epoch > 1:
        acc_improvement = acc_val - trainer.epoch_val_acc[-2]
        iou_improvement = iou_val - trainer.epoch_val_iou[-2]
        print(f"📈 Melhoria | Acc: {acc_improvement:+.2f}% | IoU: {iou_improvement:+.4f}")

def print_memory_usage():
    """Debug de uso de memória GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory: {allocated:.2f}GB alocado / {reserved:.2f}GB reservado")

if __name__=='__main__':

    # Registra o tempo de início do treinamento
    start_time = time.time()
    
    # Params
    params = {
        'root_dir': '../dataset_35/',
        'cache': True,
        'window_size': (224, 224),
        'bs': 40,
        'n_classes': 8,
        'classes': ["Urbano", "Vegetação Densa", "Sombra", "Vegetação Esparsa", "Agricultura", "Rocha", "Solo Exposto", "Água"],
        'maximum_epochs': 999,
        'save_epoch': 2,
        'print_each': 100,
        'augment': False,
        'cpu': None,
        'device': 'cuda',
        'precision': 'full',
        'optimizer_params': {
            'optimizer': 'ADAM',
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0,
            'epsilon': 1e-8,
            'momentum': 0.9
        },
        'lrs_params': {
            'type': 'Plateau',
            'lr_decay': 30,
            'milestones': [25, 35, 45],
            'gamma': 0.1
        },
        'weights': '',
        'loss': {
            'name': LossFN.TVERSKY,
            'params': {
                'weights': 'calculate',
                'alpha': 0.5,
                'gamma': 2.0,
            }
        },
        'patience': 10,
        'model': {
            'name': ModelChooser.LULC_NET,
        },
        'results_folder': "../output/LULC_NET_experiment",
    }
    
    params['results_folder'] = f"../output/LULC_NET_{params['model']['name']}_imgnet_{params['optimizer_params']['optimizer']}{params['optimizer_params']['weight_decay']}WD_{params['loss']['name']}1.0-0.5_noWeight"
    
    # Cria pasta de resultados
    os.makedirs(params['results_folder'], exist_ok=True)
    print(f"📁 Pasta de resultados criada: {params['results_folder']}")

    image_dir = os.path.join(params['root_dir'], 'images')
    label_dir = os.path.join(params['root_dir'], 'labels')
    edges_dir = os.path.join(params['root_dir'], 'edges')

    # Obtém o diretório base do projeto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)

    # Função para construir caminhos completos
    def build_full_paths(file_list, folder):
        return [os.path.join(base_dir, 'dataset_35', folder, f[0].strip()) for f in file_list]

    # Load image and label files from .txt
    print("📂 Carregando arquivos de folds...")
    train_images1 = pd.read_table('folds/fold1_images.txt', header=None).values
    train_images2 = pd.read_table('folds/fold2_images.txt', header=None).values
    train_images3 = pd.read_table('folds/fold3_images.txt', header=None).values
    train_labels1 = pd.read_table('folds/fold1_labels.txt', header=None).values
    train_labels2 = pd.read_table('folds/fold2_labels.txt', header=None).values
    train_labels3 = pd.read_table('folds/fold3_labels.txt', header=None).values

    val_images1 = pd.read_table('folds/fold4_images.txt', header=None).values
    val_labels1 = pd.read_table('folds/fold4_labels.txt', header=None).values

    test_images1 = pd.read_table('folds/fold5_images.txt', header=None).values
    test_labels1 = pd.read_table('folds/fold5_labels.txt', header=None).values

    # Constrói caminhos completos
    print("🛠️ Construindo caminhos completos...")
    train_images = build_full_paths(np.concatenate([train_images1, train_images2, train_images3]), 'images')
    train_labels = build_full_paths(np.concatenate([train_labels1, train_labels2, train_labels3]), 'labels')
    val_images = build_full_paths(val_images1, 'images')
    val_labels = build_full_paths(val_labels1, 'labels')
    test_images = build_full_paths(test_images1, 'images')
    test_labels = build_full_paths(test_labels1, 'labels')

    # Debug para verificar
    print("🔍 Verificando caminhos...")
    print(f"   ✅ Train images: {len(train_images)} arquivos")
    print(f"   ✅ Val images: {len(val_images)} arquivos")
    print(f"   ✅ Test images: {len(test_images)} arquivos")

    # Carregar os pesos de cada classe
    print("⚖️ Calculando pesos das classes...")
    weights_calculator_loss(params, train_labels)

    # Create datasets
    print("📦 Criando datasets...")
    train_dataset = DatasetIcmbio(train_images, train_labels, None, window_size=params['window_size'], cache=params['cache'], augmentation=params['augment'])
    val_dataset = DatasetIcmbio(val_images, val_labels, window_size=params['window_size'], cache=params['cache'], augmentation=False)
    test_dataset = DatasetIcmbio(test_images, test_labels, window_size=params['window_size'], cache=params['cache'], augmentation=False)

    # Create dataloaders
    print("🔄 Criando dataloaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['bs'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['bs'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['bs'], shuffle=False)

    # Build model
    print("🧠 Construindo modelo LULC-Net...")
    model = build_lulc_model(params)
    print(f"✅ Modelo {params['model']['name']} criado com sucesso!")

    loader = {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader,
    }

    # Print training summary
    print_training_summary(params, train_loader, val_loader, test_loader)

    cbkp = None
    trainer = Trainer(model, loader, params, cbkp=cbkp)
    clear()

    patCB = Callback(patience=params['patience'], min_value=60)

    print("🚀 INICIANDO TREINAMENTO!")
    print("="*80)

    # Start the training
    for epoch in range(trainer.last_epoch+1, params['maximum_epochs']):
        start_epoch_time = time.time()
        
        print(f"\n🎯 ÉPOCA {epoch}/{params['maximum_epochs']}")
        print("-"*50)

        # Training
        print("📚 Treinando...")
        acc_train, f1score_train, mcc_train, iou_train = trainer.train()

        # Validation
        print("🎯 Validando...")
        acc_val, f1score_val, mcc_val, iou_val = trainer.validate(stride=64)

        # Store metrics
        trainer.epoch_acc.append(acc_train)
        trainer.epoch_val_acc.append(acc_val)
        trainer.epoch_f1.append(f1score_train)
        trainer.epoch_val_f1.append(f1score_val)
        trainer.epoch_mcc.append(mcc_train)
        trainer.epoch_val_mcc.append(mcc_val)
        trainer.epoch_iou.append(iou_train)
        trainer.epoch_val_iou.append(iou_val)

        # Plot metrics
        trainer.plot_metrics(params['results_folder'])

        # Scheduler step
        if trainer.scheduler is not None:
            trainer.scheduler.step(iou_val)
            print(f"📉 Learning Rate atual: {trainer.scheduler.get_last_lr()[0]:.2e}")

        # Print epoch progress
        print_epoch_progress(epoch, acc_train, f1score_train, iou_train, acc_val, f1score_val, iou_val, start_epoch_time)

        # Print memory usage every 5 epochs
        if epoch % 5 == 0:
            print_memory_usage()

        # Save model if improvement
        if patCB.patience_iou_val(iou_val):
            print("💾 Salvando melhor modelo...")
            trainer.save(os.path.join(params['results_folder'], 'best_epoch.pth.tar'))

        # Early stopping
        if patCB.COUNTER == patCB.PATIENCE:
            print(f"🛑 EARLY STOPPING ativado na época {epoch}")
            print(f"🏆 Melhor época: {epoch - patCB.PATIENCE} com IoU: {patCB.BEST_VALUE:.4f}")
            break

        print("-"*50)

    # Training completed
    training_time = time.time() - start_time
    training_time_hours = training_time / 3600.0
    print(f"\n✅ TREINAMENTO CONCLUÍDO!")
    print(f"⏰ Tempo total: {training_time_hours:.2f} horas")
    print(f"📈 Melhor IoU de validação: {max(trainer.epoch_val_iou) if trainer.epoch_val_iou else 0:.4f}")

    # Save final metrics
    np.savez(os.path.join(params['results_folder'], 'metrics_train.npz'),
             acc_train=trainer.epoch_acc,
             acc_val=trainer.epoch_val_acc,
             f1score_train=trainer.epoch_f1,
             f1score_val=trainer.epoch_val_f1,
             iou_train=trainer.epoch_iou,
             iou_val=trainer.epoch_val_iou)

    # Load best model for testing
    print("\n🧪 Carregando melhor modelo para teste...")
    trainer = Trainer(model, loader, params, cbkp=os.path.join(params['results_folder'], 'best_epoch.pth.tar'))

    # Testing
    print("🔬 Executando inferências...")
    all_preds = trainer.test(stride=64, all=True)

    inference_time = time.time() - start_time - training_time
    inference_time_hours = inference_time / 3600.0
    print(f"⏰ Tempo de inferência: {inference_time_hours:.2f} horas")

    # Save inference results
    input_ids, label_ids, _ = test_loader.dataset.get_dataset()
    all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]

    os.makedirs(os.path.join(params['results_folder'], 'inference'), exist_ok=True)
    print(f"💾 Salvando {len(all_preds)} imagens de inferência...")

    for p, id_ in zip(all_preds, all_ids):
        img = convert_to_color(p)
        io.imsave(os.path.join(params['results_folder'], 'inference', f'inference_tile_{id_}.png'), img)

    print(f"\n🎉 PROCESSO COMPLETO!")
    print(f"📁 Resultados salvos em: {params['results_folder']}")