def read_zip_binary(path):
    """Read binary data from zip file"""
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split("/")[-1].split(".")[0]
        print(f"Reading {inner_path}.bin from {path}")
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array, r)
    return ragged_array

def read_binary(path):
    """Read binary data from unzipped file"""
    ragged_array = []
    with open(path, "rb") as r:
        read_binary_from(ragged_array, r)
    return ragged_array

def read_binary_from(ragged_array, r):
    """Helper function to read binary data"""
    while True:
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

def load_labels(path):
    """Load labels from CSV file"""
    df = pd.read_csv(path, header=None)
    return df[0].values


class ECGDataExplorer:
    def __init__(self, X_data, y_labels):
        self.X_data = X_data
        self.y_labels = y_labels
        self.class_names = ['Normal', 'AF', 'Other', 'Noisy']

    def analyze_data(self):
        """Perform data analysis"""
        print("=== ECG Data Analysis ===")

        lengths = [len(x) for x in self.X_data]
        print(f"Total samples: {len(self.X_data)}")
        print(f"Length statistics:")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.1f}, Std: {np.std(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")

        unique, counts = np.unique(self.y_labels, return_counts=True)
        print(f"\nClass distribution:")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            print(f"  Class {cls} ({self.class_names[cls]}): {count} ({count/len(self.y_labels)*100:.1f}%)")

        return {
            'lengths': lengths,
            'class_distribution': dict(zip(unique, counts))
        }

    def visualize_samples(self):
        """Visualize sample ECG signals from each class"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for class_idx in range(4):
            class_indices = np.where(self.y_labels == class_idx)[0]
            if len(class_indices) > 0:
                sample_idx = class_indices[0]
                signal_data = self.X_data[sample_idx]

                axes[class_idx].plot(signal_data)
                axes[class_idx].set_title(f'Class {class_idx}: {self.class_names[class_idx]}')
                axes[class_idx].set_xlabel('Time (samples)')
                axes[class_idx].set_ylabel('Amplitude')
                axes[class_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_validation_split(self, test_size=0.2):
        """Create stratified validation split"""
        indices = np.arange(len(self.X_data))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, stratify=self.y_labels, random_state=42
        )

        print(f"\nValidation split created:")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")

        train_classes = np.bincount(self.y_labels[train_idx])
        val_classes = np.bincount(self.y_labels[val_idx])

        train_total = train_classes.sum()
        val_total = val_classes.sum()

        train_percent = train_classes / train_total * 100
        val_percent = val_classes / val_total * 100

        print("  Training class distribution:")
        for i, (count, percent) in enumerate(zip(train_classes, train_percent)):
            print(f"    Class {i}: {count} ({percent:.1f}%)")

        print("  Validation class distribution:")
        for i, (count, percent) in enumerate(zip(val_classes, val_percent)):
            print(f"    Class {i}: {count} ({percent:.1f}%)")


        return train_idx, val_idx
		

if os.path.exists('./Uni/AMLS'):
  X_train = read_zip_binary('./Uni/AMLS/X_train.zip')
  y_train = load_labels('./Uni/AMLS/y_train.csv')
  X_test = read_zip_binary('./Uni/AMLS/X_test.zip')
else:
  X_train = read_zip_binary('/content/drive/MyDrive/AMLS/X_train.zip')
  y_train = load_labels('/content/drive/MyDrive/AMLS/y_train.csv')
  X_test = read_zip_binary('/content/drive/MyDrive/AMLS/X_test.zip')


print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
print("\n" + " TASK 1: DATA EXPLORATION ")

explorer = ECGDataExplorer(X_train, y_train)
stats = explorer.analyze_data()
explorer.visualize_samples()
train_idx, val_idx = explorer.create_validation_split()

X_train_split = [X_train[i] for i in train_idx]
y_train_split = y_train[train_idx]
X_val_split = [X_train[i] for i in val_idx]
y_val_split = y_train[val_idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")