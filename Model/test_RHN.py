"""
Script de test des architectures RecurentAttentionNetwork & RecurentSpectralLayer
Compatible avec la version optimisée production.
"""

import torch
import torch.nn as nn

from .RecurentHyperNetwork.RecurentSpectralLayer import RecurentSpectralLayer, RecurentBilineaireSpectralLayer
from .RecurentHyperNetwork.RecurentAttentionNetwork import RecurentAttentionNetwork


def test_spectral_layer():
    print("\n" + "="*60)
    print("TEST: RecurentSpectralLayer")
    print("="*60)

    in_dim, out_dim = 10, 20
    batch_size = 5

    layer = RecurentSpectralLayer(in_dim, out_dim)
    x = torch.randn(batch_size, in_dim)

    output = layer(x)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")

    assert output.shape == (batch_size, out_dim), \
        f"Shape attendu ({batch_size}, {out_dim}), obtenu {output.shape}"
    print("✓ Test de dimensions: PASSED")

    output.sum().backward()
    assert layer.W_bc.grad is not None, "Pas de gradient pour W_bc"
    assert layer.W_ew.grad is not None, "Pas de gradient pour W_ew"
    print("✓ Test de gradient: PASSED")

    for bs in [1, 10, 32]:
        out_test = layer(torch.randn(bs, in_dim))
        assert out_test.shape == (bs, out_dim)
    print("✓ Test différentes tailles de batch: PASSED")

    print("\n✅ RecurentSpectralLayer: TOUS LES TESTS PASSED\n")


def test_bilineare_spectral_layer():
    print("\n" + "="*60)
    print("TEST: RecurentBilineaireSpectralLayer")
    print("="*60)

    in_dim, out_dim = 15, 25
    batch_size = 8

    layer = RecurentBilineaireSpectralLayer(in_dim, out_dim)
    x = torch.randn(batch_size, in_dim)

    output = layer(x)

    assert output.shape == (batch_size, out_dim), \
        f"Shape attendu ({batch_size}, {out_dim}), obtenu {output.shape}"
    print("✓ Test de dimensions: PASSED")

    output.sum().backward()
    assert layer.W_proj.grad is not None, "Pas de gradient pour W_proj"
    assert layer.W_ew.grad is not None,   "Pas de gradient pour W_ew"
    print("✓ Test de gradient: PASSED")

    # Compatibilité de shape avec la couche standard
    layer_std = RecurentSpectralLayer(in_dim, out_dim)
    assert layer_std(x).shape == output.shape
    print("✓ Test compatibilité des dimensions: PASSED")

    print("\n✅ RecurentBilineaireSpectralLayer: TOUS LES TESTS PASSED\n")


def test_attention_hypernetwork_basic():
    print("\n" + "="*60)
    print("TEST: RecurentAttentionNetwork (Basic)")
    print("="*60)

    layers_dim = [10, 20, 30, 15]
    batch_size = 4

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=0.0)
    x = torch.randn(batch_size, layers_dim[0])

    output = network.forward(x)

    expected_shape = (batch_size, layers_dim[-1])
    assert output.shape == expected_shape, \
        f"Shape attendu {expected_shape}, obtenu {output.shape}"
    print(f"✓ Test de dimensions: PASSED — {output.shape}")

    # Compatibilité avec test_AHN : network.layers contient spectral + norms
    # Nombre attendu = (n_layers - 1) spectral + (n_layers - 2) norms (pas de norm sur dernière couche)
    n_spectral = len(layers_dim) - 1
    n_norms = len(layers_dim) - 2  # use_layernorm=True mais pas sur dernière couche
    expected_layers = n_spectral + n_norms
    actual = len(network.layers)
    print(f"✓ Nombre de couches dans network.layers: {actual} (attendu: {expected_layers})")
    assert actual == expected_layers, f"Attendu {expected_layers}, obtenu {actual}"
    print("✓ Test du nombre de couches: PASSED")

    print("\n✅ RecurentAttentionNetwork (Basic): TOUS LES TESTS PASSED\n")


def test_attention_hypernetwork_without_layernorm():
    print("\n" + "="*60)
    print("TEST: RecurentAttentionNetwork (Sans LayerNorm)")
    print("="*60)

    layers_dim = [5, 10, 8]
    batch_size = 3

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=False, dropout=0.0)
    x = torch.randn(batch_size, layers_dim[0])

    output = network.forward(x)

    expected_shape = (batch_size, layers_dim[-1])
    assert output.shape == expected_shape

    # Sans LayerNorm : seulement les couches spectrales dans network.layers
    expected_layers = len(layers_dim) - 1
    actual = len(network.layers)
    assert actual == expected_layers, f"Attendu {expected_layers}, obtenu {actual}"
    print(f"✓ Nombre de couches: {actual} (attendu: {expected_layers})")
    print("✓ Test sans LayerNorm: PASSED")

    print("\n✅ RecurentAttentionNetwork (Sans LayerNorm): TOUS LES TESTS PASSED\n")


def test_attention_hypernetwork_with_dropout():
    print("\n" + "="*60)
    print("TEST: RecurentAttentionNetwork (Avec Dropout)")
    print("="*60)

    layers_dim = [8, 16, 8]
    batch_size = 10
    dropout = 0.5

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=dropout)
    x = torch.randn(batch_size, layers_dim[0])

    network.dropout.train()
    output_train = network.forward(x)

    network.dropout.eval()
    output_eval = network.forward(x)

    expected_shape = (batch_size, layers_dim[-1])
    assert output_train.shape == expected_shape, "Erreur shape en mode train"
    assert output_eval.shape  == expected_shape, "Erreur shape en mode eval"
    print(f"✓ Output shapes train/eval: {output_train.shape}")
    print("✓ Test avec dropout: PASSED")

    print("\n✅ RecurentAttentionNetwork (Avec Dropout): TOUS LES TESTS PASSED\n")


def test_deep_network():
    print("\n" + "="*60)
    print("TEST: Réseau Profond")
    print("="*60)

    layers_dim = [5, 10, 20, 30, 20, 10, 5]
    batch_size = 2

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=0.1)
    x = torch.randn(batch_size, layers_dim[0])

    output = network.forward(x)

    expected_shape = (batch_size, layers_dim[-1])
    assert output.shape == expected_shape
    print(f"✓ Configuration profonde: {layers_dim}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Test réseau profond: PASSED")

    print("\n✅ Réseau Profond: TOUS LES TESTS PASSED\n")


def test_gradient_flow():
    print("\n" + "="*60)
    print("TEST: Flux de Gradient")
    print("="*60)

    layers_dim = [10, 20, 15]
    batch_size = 5

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=0.0)
    x = torch.randn(batch_size, layers_dim[0], requires_grad=True)

    output = network.forward(x)
    output.sum().backward()

    assert x.grad is not None, "Pas de gradient pour l'input"
    print("✓ Gradient de l'input: OK")

    spectral_layers = [l for l in network.layers if isinstance(l, RecurentSpectralLayer)]
    for i, layer in enumerate(spectral_layers):
        assert layer.W_bc.grad is not None, f"Pas de gradient pour layer {i} W_bc"
        assert layer.W_ew.grad is not None, f"Pas de gradient pour layer {i} W_ew"
    print(f"✓ Gradients des {len(spectral_layers)} couches spectrales: OK")

    print("✓ Test du flux de gradient: PASSED")
    print("\n✅ Flux de Gradient: TOUS LES TESTS PASSED\n")


def test_numerical_stability():
    print("\n" + "="*60)
    print("TEST: Stabilité Numérique")
    print("="*60)

    layers_dim = [10, 50, 100, 50, 10]
    batch_size = 8

    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=0.0)

    x_normal = torch.randn(batch_size, layers_dim[0])
    out_normal = network.forward(x_normal)
    assert not torch.isnan(out_normal).any(), "NaN avec input normal"
    assert not torch.isinf(out_normal).any(), "Inf avec input normal"
    print("✓ Input normal: OK")

    x_large = torch.randn(batch_size, layers_dim[0]) * 10
    out_large = network.forward(x_large)
    assert not torch.isnan(out_large).any(), "NaN avec grandes valeurs"
    assert not torch.isinf(out_large).any(), "Inf avec grandes valeurs"
    print("✓ Grandes valeurs (×10): OK")

    print("✓ Test de stabilité numérique: PASSED")
    print("\n✅ Stabilité Numérique: TOUS LES TESTS PASSED\n")


def test_gradient_checkpointing():
    print("\n" + "="*60)
    print("TEST: Gradient Checkpointing")
    print("="*60)

    layers_dim = [10, 20, 15]
    batch_size = 4

    network = RecurentAttentionNetwork(
        layers_dim, use_layernorm=True, dropout=0.0, use_checkpointing=True
    )
    network.train()

    x = torch.randn(batch_size, layers_dim[0], requires_grad=True)
    output = network.forward(x)
    output.sum().backward()

    assert x.grad is not None, "Pas de gradient avec checkpointing"
    print("✓ Gradient checkpointing: OK")
    print("\n✅ Gradient Checkpointing: TOUS LES TESTS PASSED\n")


def test_parameter_count():
    print("\n" + "="*60)
    print("TEST: Comptage des Paramètres")
    print("="*60)

    layers_dim = [10, 20, 15]
    network = RecurentAttentionNetwork(layers_dim, use_layernorm=True, dropout=0.0)
    n_params = network.count_parameters()
    print(f"✓ Paramètres totaux: {n_params:,}")
    assert n_params > 0
    print("✓ Test comptage paramètres: PASSED")
    print("\n✅ Comptage Paramètres: TOUS LES TESTS PASSED\n")


def run_all_tests():
    print("\n" + "="*60)
    print("DÉBUT DES TESTS — RecurentAttentionNetwork & RecurentSpectralLayer")
    print("="*60)

    try:
        test_spectral_layer()
        test_bilineare_spectral_layer()
        test_attention_hypernetwork_basic()
        test_attention_hypernetwork_without_layernorm()
        test_attention_hypernetwork_with_dropout()
        test_deep_network()
        test_gradient_flow()
        test_numerical_stability()
        test_gradient_checkpointing()
        test_parameter_count()

        print("\n" + "="*60)
        print("🎉 TOUS LES TESTS ONT RÉUSSI! 🎉")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERREUR DÉTECTÉE: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
