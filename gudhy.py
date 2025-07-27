import gudhi
import numpy as np
import ot
from gudhi.wasserstein import wasserstein_distance
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw as dtw_viz

# Chargement des matrices de similarité
a = np.load('cost_matrices_output/gt.npy')
b = np.load('cost_matrices_output/predicted.npy')
c = np.load('cost_matrices_output/noisy.npy')

# Pour la Vérité Terrain
rips_complex_gt = gudhi.RipsComplex(distance_matrix=a, max_edge_length=1.0) # max_edge_length est le rayon max de la filtration
simplex_tree_gt = rips_complex_gt.create_simplex_tree(max_dimension=1) # Calcule H0 (connexité) et H1 (trous)
diagram_gt = simplex_tree_gt.persistence()
print("diagramme de persistance VT:")
for dim, (birth, death) in diagram_gt:
    print(f"Dim {dim}: [{birth:.3f}, {death:.3f})")

# Pour les données prédites 
rips_complex_pred = gudhi.RipsComplex(distance_matrix=b, max_edge_length=1.0)
simplex_tree_pred = rips_complex_pred.create_simplex_tree(max_dimension=1)
diagram_pred = simplex_tree_pred.persistence()
print("\n\n diagramme de persistance PRED:")
for dim, (birth, death) in diagram_pred:
    print(f"Dim {dim}: [{birth:.3f}, {death:.3f})")

# Extraction des diagrammes par dimension
diagram_H0_gt = np.array([pt[1] for pt in diagram_gt if pt[0] == 0])
diagram_H0_pred = np.array([pt[1] for pt in diagram_pred if pt[0] == 0])

dist_bottleneck_H0 = gudhi.bottleneck_distance(
        diagram_H0_gt, diagram_H0_pred
    )
print("distance de bottleneck : ", dist_bottleneck_H0)

# Nombre d'entrées dans la vt
n = float(a.shape[0])

# Calculer ls distances entre les diagrammes
# Bottleneck distance pour H0
if len(diagram_H0_gt) > 0 and len(diagram_H0_pred) > 0:
    dist_bottleneck_H0 = gudhi.bottleneck_distance(
        diagram_H0_gt, diagram_H0_pred
    )
    print(f"\nDistance de Bottleneck (H0) : {dist_bottleneck_H0:.4f}")
else:
    print("\nPas assez de points en H0 pour calculer la distance de Bottleneck.")

# Wasserstein distance pour H0 (qualité globale)
if len(diagram_H0_gt) > 0 and len(diagram_H0_pred) > 0:
    dist_wasserstein_H0 = wasserstein_distance(
        diagram_H0_gt, diagram_H0_pred, order=1.0, internal_p=2.0
    )
    print(f"Distance de Wasserstein (H0): {1 - (dist_wasserstein_H0 / n)}")
else:
    print("Pas assez de points en H0 pour calculer la distance de Wasserstein.")

max_finite_val_H0 = 0
max_finite_val_H0 = max(max_finite_val_H0, np.max(diagram_H0_gt[np.isfinite(diagram_H0_gt)]))
max_finite_val_H0 = max(max_finite_val_H0, np.max(diagram_H0_pred[np.isfinite(diagram_H0_pred)]))

n_steps = 100 # Nombre de points sur la courbe
filtration_values = np.linspace(0, max_finite_val_H0 * 1.1, n_steps) 

# Fonction pour calculer une courbe de Betti à partir d'un diagramme
def compute_betti_curve(diagram, filtration_values):
    betti_curve = []
    for t in filtration_values:
        count = 0
        for birth, death in diagram:
            if birth <= t and death > t:
                count += 1
        betti_curve.append(count)
    return np.array(betti_curve)

# Calculer les courbes de Betti pour H0
betti_H0_gt = compute_betti_curve(diagram_H0_gt, filtration_values)
betti_H0_pred = compute_betti_curve(diagram_H0_pred, filtration_values)

# Afficher les courbes de Betti H0
plt.figure(figsize=(8, 6))
plt.plot(filtration_values, betti_H0_gt, label="H0 Ground Truth", color='blue')
plt.plot(filtration_values, betti_H0_pred, label="H0 Prediction", color='red', linestyle='--')
plt.xlabel("Valeur de Filtration (epsilon)")
plt.ylabel("Nombre de Composantes Connexes (Betti 0)")
plt.title("Courbes de Betti (H0)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Longueur de betti_H0_gt: {len(betti_H0_gt)}")
print(f"Longueur de betti_H0_pred: {len(betti_H0_pred)}")

# Calcul du DTW avec fastdtw 
distance_dtw, path = fastdtw(betti_H0_gt, betti_H0_pred, dist=lambda u, v: abs(u-v))
print(f"\nLa distance DTW entre les courbes H0 Ground Truth et H0 Prediction est : {distance_dtw}")


alignment = dtw_viz(betti_H0_gt, betti_H0_pred, keep_internals=True)
alignment.plot(type="threeway")
plt.title("Alignement DTW des Courbes de Betti H0")
plt.show()

betti_H0_pire_cas = np.full(len(betti_H0_gt), 1.0) # On utilise 1.0 pour s'assurer que c'est un flottant si besoin
distance_dtw_gt_vs_pire_cas, path = fastdtw(betti_H0_gt, betti_H0_pire_cas, dist=lambda u, v: abs(u - v))
print(f"\nLa distance DTW entre la courbe H0 Ground Truth et la courbe 'pire cas' (y=1) est : {distance_dtw_gt_vs_pire_cas}")
score = 1 - (distance_dtw / distance_dtw_gt_vs_pire_cas)
print(score)

exit()