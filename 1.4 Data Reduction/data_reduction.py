class ECGFeatureExtractor:
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def extract_statistical_features(self, signal):
        """Extract basic statistical features"""
        features = []
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            len(signal)
        ])
        return features

    def extract_frequency_features(self, signal):
        """Extract frequency domain features"""
        # Compute FFT
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]

        features = []
        features.extend([
            np.mean(positive_fft),
            np.std(positive_fft),
            np.max(positive_fft),
            np.argmax(positive_fft),  # Dominant frequency index
        ])

        return features

    def extract_all_features(self, signals):
        """Extract all features for a list of signals"""
        all_features = []
        for signal in signals:
            features = []
            features.extend(self.extract_statistical_features(signal))
            features.extend(self.extract_frequency_features(signal))
            all_features.append(features)

        return np.array(all_features)


class ECGDataReducer:
    def __init__(self):
        pass

    def stratified_sample(self, X_data, y_labels, reduction_ratio=0.25):
        """Stratified random sampling"""
        indices = np.arange(len(X_data))
        _, sampled_indices = train_test_split(
            indices, test_size=reduction_ratio, stratify=y_labels, random_state=42
        )

        return [X_data[i] for i in sampled_indices], y_labels[sampled_indices]

    def coreset_selection(self, X_data, y_labels, reduction_ratio=0.25):
        """Simple coreset selection based on feature diversity"""
        # Extract features for coreset selection
        feature_extractor = ECGFeatureExtractor()
        features = feature_extractor.extract_all_features(X_data)

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Select diverse samples per class
        selected_indices = []
        unique_classes = np.unique(y_labels)

        for cls in unique_classes:
            class_indices = np.where(y_labels == cls)[0]
            class_features = features_normalized[class_indices]

            # Select samples with highest feature variance
            n_select = max(1, int(len(class_indices) * reduction_ratio))

            if len(class_indices) <= n_select:
                selected_indices.extend(class_indices)
            else:
                # Use k-means-like selection
                from sklearn.cluster import KMeans
                if n_select > 1:
                    kmeans = KMeans(n_clusters=n_select, random_state=42)
                    kmeans.fit(class_features)

                    # Select samples closest to centroids
                    for center in kmeans.cluster_centers_:
                        distances = np.linalg.norm(class_features - center, axis=1)
                        closest_idx = class_indices[np.argmin(distances)]
                        if closest_idx not in selected_indices:
                            selected_indices.append(closest_idx)
                else:
                    # If only selecting one sample, pick the one closest to mean
                    mean_features = np.mean(class_features, axis=0)
                    distances = np.linalg.norm(class_features - mean_features, axis=1)
                    selected_indices.append(class_indices[np.argmin(distances)])

        return [X_data[i] for i in selected_indices], y_labels[selected_indices]

    def compress_signal(self, signal, compression_ratio=0.5):
        """Simple signal compression using FFT"""
        fft_signal = fft(signal)
        n_keep = int(len(fft_signal) * compression_ratio)

        # Keep low frequency components
        compressed_fft = np.zeros_like(fft_signal)
        compressed_fft[:n_keep//2] = fft_signal[:n_keep//2]
        compressed_fft[-n_keep//2:] = fft_signal[-n_keep//2:]

        return np.real(ifft(compressed_fft))


def compress_dataset(X_data, y_labels, output_path="compressed_data.zip"):
    """Compress dataset using custom format"""

    binary_data = bytearray()
    for signal in X_data:
        binary_data.extend(struct.pack('i', len(signal)))
        for val in signal:
            binary_data.extend(struct.pack('h', int(val)))
    labels_df = pd.DataFrame({'label': y_labels})
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('data.bin', binary_data)
        zf.writestr('labels.csv', labels_df.to_csv(index=False))

    print(f"Compressed dataset saved to {output_path}")
    print(f"Original size: {len(binary_data)} bytes")
    print(f"Compressed size: {os.path.getsize(output_path)} bytes")
    print(f"Compression ratio: {os.path.getsize(output_path)/len(binary_data):.3f}")


def train_and_evaluate_reduced_models(X_train_split, y_train_split, X_val_split, y_val_split,
                                    X_test, device='cpu'):
    """
    Train models on reduced datasets and evaluate performance
    """
    reducer = ECGDataReducer()
    reduction_ratios = [0.1, 0.25, 0.5, 1.0] 
    results = {
        'reduction_ratio': [],
        'method': [],
        'f1_score': [],
        'accuracy': [],
        'train_samples': [],
        'val_f1': [],
        'val_accuracy': []
    }

    print("Training models on reduced datasets...")
    print("=" * 60)

    for ratio in reduction_ratios:
        print(f"\nTraining with {ratio*100}% of data...")

        if ratio == 1.0:
            X_train_reduced = X_train_split
            y_train_reduced = y_train_split
            method = 'full'
        else:
            for method in ['stratified', 'coreset']:
                print(f"  Method: {method}")

                if method == 'stratified':
                    X_train_reduced, y_train_reduced = reducer.stratified_sample(
                        X_train_split, y_train_split, ratio
                    )
                else: 
                    X_train_reduced, y_train_reduced = reducer.coreset_selection(
                        X_train_split, y_train_split, ratio
                    )
                print(f"    Training samples: {len(X_train_reduced)}")
                model, train_history = train_model_on_reduced_data(
                    X_train_reduced, y_train_reduced, X_val_split, y_val_split,
                    device, f"reduced_{ratio}_{method}"
                )
                val_f1, val_acc = evaluate_model_on_validation(
                    model, X_val_split, y_val_split, device
                )
                results['reduction_ratio'].append(ratio)
                results['method'].append(method)
                results['f1_score'].append(train_history['best_f1'])
                results['accuracy'].append(val_acc)
                results['train_samples'].append(len(X_train_reduced))
                results['val_f1'].append(val_f1)
                results['val_accuracy'].append(val_acc)

                print(f"    Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
                if ratio == 0.25 and method == 'coreset':
                    print("    Generating predictions for reduced.csv...")
                    generate_reduced_predictions(model, X_test, device)

        if ratio == 1.0:
            print(f"  Method: full dataset")
            model, train_history = train_model_on_reduced_data(
                X_train_reduced, y_train_reduced, X_val_split, y_val_split,
                device, "full"
            )

            val_f1, val_acc = evaluate_model_on_validation(
                model, X_val_split, y_val_split, device
            )

            results['reduction_ratio'].append(ratio)
            results['method'].append('full')
            results['f1_score'].append(train_history['best_f1'])
            results['accuracy'].append(val_acc)
            results['train_samples'].append(len(X_train_reduced))
            results['val_f1'].append(val_f1)
            results['val_accuracy'].append(val_acc)

            print(f"    Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")

    return results

def train_model_on_reduced_data(X_train, y_train, X_val, y_val, device, name_prefix):
    """
    Train a model on reduced data
    """
    augmentor = ECGAugmentor()
    train_dataset = ECGDataset(X_train, y_train, augmentor=augmentor, is_training=True)
    val_dataset = ECGDataset(X_val, y_val, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model = ImprovedSTFTModel(num_classes=4)
    trainer = ECGTrainer(model, device=device, name_prefix=name_prefix + '_')
    history = trainer.train(train_loader, val_loader, epochs=80, patience=40)

    return model, history

def evaluate_model_on_validation(model, X_val, y_val, device):
    """
    Evaluate model on validation set
    """
    val_dataset = ECGDataset(X_val, y_val, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets, lengths in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data, lengths)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')
    accuracy = accuracy_score(all_targets, all_preds)

    return f1, accuracy

def generate_reduced_predictions(model, X_test, device):
    """
    Generate predictions for the test set using reduced model
    """
    test_dataset = ECGDataset(X_test, [0] * len(X_test), is_training=False)  # Dummy labels
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    predictions = []

    with torch.no_grad():
        for data, _, lengths in test_loader:
            data = data.to(device)
            outputs = model(data, lengths)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    output_path = 'reduced.csv'
    if os.path.exists('./Uni/AMLS'):
        output_path = './Uni/AMLS/reduced.csv'

    pd.DataFrame({'label': predictions}).to_csv(output_path, index=False)
    print(f"Reduced predictions saved to {output_path}")

def create_results_plot(results):
    """
    Create visualization of results across different data sizes
    """

    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for method in ['stratified', 'coreset', 'full']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax1.plot(method_data['reduction_ratio'] * 100, method_data['val_f1'],
                    marker='o', linewidth=2, markersize=8, label=method.capitalize())

    ax1.set_xlabel('Dataset Size (%)')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)

    for method in ['stratified', 'coreset', 'full']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax2.plot(method_data['reduction_ratio'] * 100, method_data['val_accuracy'],
                    marker='s', linewidth=2, markersize=8, label=method.capitalize())

    ax2.set_xlabel('Dataset Size (%)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig('data_reduction_results.png', dpi=300, bbox_inches='tight')
    if os.path.exists('./Uni/AMLS'):
        plt.savefig('./Uni/AMLS/data_reduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("SUMMARY OF DATA REDUCTION RESULTS")
    print("="*60)

    summary_df = df.pivot_table(
        values=['val_f1', 'val_accuracy'],
        index='reduction_ratio',
        columns='method',
        aggfunc='mean'
    )

    print("\nF1 Scores:")
    print(summary_df['val_f1'].round(4))
    print("\nAccuracy:")
    print(summary_df['val_accuracy'].round(4))

    print("\n" + "="*60)
    print("CORESET vs STRATIFIED COMPARISON")
    print("="*60)

    for ratio in [0.1, 0.25, 0.5]:
        strat_f1 = df[(df['reduction_ratio'] == ratio) & (df['method'] == 'stratified')]['val_f1'].values
        coreset_f1 = df[(df['reduction_ratio'] == ratio) & (df['method'] == 'coreset')]['val_f1'].values

        if len(strat_f1) > 0 and len(coreset_f1) > 0:
            improvement = coreset_f1[0] - strat_f1[0]
            print(f"{ratio*100:2.0f}% data: Coreset F1={coreset_f1[0]:.4f}, Stratified F1={strat_f1[0]:.4f}, "
                  f"Improvement={improvement:+.4f}")

def create_compressed_datasets(X_train_split, y_train_split, reduction_ratios=[0.1, 0.25, 0.5]):
    """
    Create compressed datasets for different reduction ratios
    """

    reducer = ECGDataReducer()

    print("Creating compressed datasets...")
    print("=" * 40)

    for ratio in reduction_ratios:
        X_reduced, y_reduced = reducer.coreset_selection(X_train_split, y_train_split, ratio)

        output_path = f'reduced_data_{int(ratio*100)}percent.zip'
        if os.path.exists('./Uni/AMLS'):
            output_path = f'./Uni/AMLS/reduced_data_{int(ratio*100)}percent.zip'

        compress_dataset(X_reduced, y_reduced, output_path)

        print(f"Created {output_path} with {len(X_reduced)} samples")

def read_compressed_dataset(path):
    """
    Read compressed dataset
    """
    import zipfile
    import struct

    X_data = []

    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = os.path.basename(path).split('.')[0]

        try:
            with zf.open('data.bin', 'r') as f:
                read_binary_from(X_data, f)
        except:
            with zf.open(f'{inner_path}.bin', 'r') as f:
                read_binary_from(X_data, f)

        try:
            with zf.open('labels.csv', 'r') as f:
                labels_df = pd.read_csv(f)
                y_labels = labels_df['label'].values
        except:
            with zf.open(f'{inner_path}_labels.csv', 'r') as f:
                labels_df = pd.read_csv(f)
                y_labels = labels_df['label'].values

    return X_data, y_labels
    
print("\n" + " TASK: DATA REDUCTION ")

results = train_and_evaluate_reduced_models(
    X_train_split, y_train_split, X_val_split, y_val_split, X_test, device
)

create_results_plot(results)
create_compressed_datasets(X_train_split, y_train_split)
results_df = pd.DataFrame(results)
output_path = '/content/drive/MyDrive/AMLS/data_reduction_results.csv'
if os.path.exists('./Uni/AMLS'):
    output_path = './Uni/AMLS/data_reduction_results.csv'

results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

print("\n" + "="*60)
print("\n" + "="*60)
print("DATA REDUCTION EXPERIMENT COMPLETED")
print("="*60)
print("Files created:")
print("** reduced.csv: Test predictions using 25% coreset model")
print("** data_reduction_results.png: Performance comparison plot")
print("** data_reduction_results.csv: Detailed results")
print("** reduced_data_*percent.zip: Compressed datasets")