import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np

# -------------------------------
# Model
# -------------------------------
class GrapeGAT(torch.nn.Module):
    def __init__(self, in_dim, hid=128, out=2, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid * heads, hid, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hid * heads, hid, heads=1, dropout=dropout)

        self.bn1 = torch.nn.BatchNorm1d(hid * heads)
        self.bn2 = torch.nn.BatchNorm1d(hid * heads)
        self.bn3 = torch.nn.BatchNorm1d(hid)

        self.fc1 = torch.nn.Linear(hid * 3 + 5, hid)
        self.fc2 = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch, graph_feats):
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        x3 = F.elu(self.bn3(self.conv3(x2, edge_index)))

        # Residual connection
        x = x3 + x1[:, :x3.size(1)]

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)

        # CORRECT CONCAT
        x = torch.cat([x_mean, x_max, x_add, graph_feats], dim=1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.fc2(x)


# -------------------------------
# Graph Features
# -------------------------------
def compute_graph_features(g, edges):
    num_nodes = len(g)
    num_edges = len(edges)

    degree = np.zeros(num_nodes)
    for e in edges:
        degree[e[0]] += 1

    avg_degree = degree.mean() if num_nodes > 0 else 0
    junction_ratio = (g['type'] == 'junction').mean() if 'type' in g.columns else 0
    avg_width = g['width'].mean() if 'width' in g.columns else 0

    return [
        num_nodes / 200,
        num_edges / 400,
        avg_degree / 10,
        junction_ratio,
        avg_width / 5
    ]


# -------------------------------
# Load Graphs
# -------------------------------
def load_graphs(graph_path, label_path=None):
    df = pd.read_csv(graph_path)
    labels = pd.read_csv(label_path) if label_path else None

    graphs = []

    for gid in df['graph_id'].unique():
        g = df[df['graph_id'] == gid].reset_index(drop=True)

        node_type_map = {'junction': 0, 'endpoint': 1}
        type_vals = g['type'].map(lambda t: node_type_map.get(t, 0)).values if 'type' in g.columns else np.zeros(len(g))

        x = np.column_stack([
            g['x'].values / 600,
            g['y'].values / 600,
            g['width'].values / 20,
            type_vals,
        ])
        x = torch.tensor(x, dtype=torch.float)

        edges = []
        node_map = {row['node_id']: i for i, row in g.iterrows()}

        for i, row in g.iterrows():
            if pd.notna(row['edges']) and row['edges']:
                for tgt in str(row['edges']).split(';'):
                    if tgt.strip().isdigit():
                        tgt_id = int(tgt)
                        if tgt_id in node_map:
                            j = node_map[tgt_id]
                            edges.append([i, j])
                            edges.append([j, i])  # bidirectional

        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)

        graph_feats = compute_graph_features(g, edges)

        y = torch.tensor([labels[labels['graph_id'] == gid]['label'].values[0]]) if labels is not None else None

        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid

        # ✅ FIXED SHAPE
        data.graph_feats = torch.tensor(graph_feats, dtype=torch.float).unsqueeze(0)

        graphs.append(data)

    return graphs


# -------------------------------
# Training
# -------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_graphs('data/public/train_data.csv', 'data/public/train_labels.csv')

    # Use MOST data for training (small dataset)
    train_graphs = graphs[:-5]
    val_graphs = graphs[-5:]

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(labels)

    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(device)

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16)

    model = GrapeGAT(in_dim=4).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0
    patience = 50
    patience_counter = 0
    best_model_state = None

    for ep in range(500):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            graph_feats = batch.graph_feats  # ✅ FIXED

            opt.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
                loss = F.cross_entropy(out, batch.y, weight=class_weights, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += len(batch.y)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                graph_feats = batch.graph_feats

                out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
                pred = out.argmax(dim=1)

                val_correct += (pred == batch.y).sum().item()
                val_total += len(batch.y)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}: loss={total_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(best_model_state)
    print(f"Best validation accuracy: {best_val_acc:.3f}")

    return model


# -------------------------------
# Prediction
# -------------------------------
def predict(model, graph_path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_graphs(graph_path)
    loader = DataLoader(graphs, batch_size=16)

    preds = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            graph_feats = batch.graph_feats  # ✅ FIXED

            out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
            pred = out.argmax(dim=1).cpu().numpy()

            for i, gid in enumerate(batch.gid):
                preds.append({'graph_id': gid, 'label': int(pred[i])})

    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    model = train()
    predict(model, 'data/public/test_data.csv', 'submission.csv')
