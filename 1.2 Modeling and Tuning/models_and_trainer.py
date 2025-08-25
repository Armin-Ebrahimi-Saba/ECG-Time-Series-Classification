class SimpleSTFTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # STFT parameters
        self.n_fft = 1024
        self.hop_length = 128

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            self._conv_block(1, 64, dropout=0.3),
            self._conv_block(64, 128, dropout=0.3),
            self._conv_block(128, 256, dropout=0.4),
            self._conv_block(256, 512, dropout=0.4),
            self._conv_block(512, 256, dropout=0.4)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # RNNs (lazy initialized)
        self.rnn1 = self.rnn2 = None
        self.rnn_input_size = None

        # Fully connected classifier
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )

    def _compute_stft_batch(self, x):
        """Compute STFT for each sample individually and pad to uniform size"""
        batch_size = x.size(0)
        device = x.device
        stft_outputs = []

        for i in range(batch_size):
            signal_data = x[i].detach().cpu().numpy()
            _, _, stft = signal.stft(signal_data, 
                                     nperseg=self.n_fft // 4, 
                                     noverlap=self.n_fft // 8,
                                     window='hann')
            stft_mag = np.abs(stft)
            stft_outputs.append(torch.tensor(stft_mag, dtype=torch.float32))

        max_freq = max(s.size(0) for s in stft_outputs)
        max_time = max(s.size(1) for s in stft_outputs)
        padded = torch.zeros(batch_size, 1, max_freq, max_time, device=device)

        for i, s in enumerate(stft_outputs):
            s = s.to(device)
            padded[i, 0, :s.size(0), :s.size(1)] = s

        return torch.log2(padded + 1e-10)

    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        device = x.device

        # STFT → Conv → Pool
        x = self._compute_stft_batch(x)
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)

        # Reshape for RNN
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, x.size(3), -1)

        # Lazy RNN init
        if self.rnn1 is None:
            self.rnn_input_size = x.size(-1)
            self.rnn1 = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True).to(device)
            self.rnn2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True).to(device)
            print(f"Initialized LSTM layers with input size: {self.rnn_input_size}")

        # BiLSTM layers
        x, _ = self.rnn1(x)
        x = self.dropout2(x)
        x, (h2, _) = self.rnn2(x)

        # Concatenate forward and backward final states
        x = torch.cat([h2[0], h2[1]], dim=1)  # shape: (batch, 512)

        # Fully connected classifier
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)

        return self.fc4(x)
        
        
class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, V)
        pooled = attended.mean(dim=1)
        return pooled, attn_weights


class ImprovedSTFTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.n_fft = 512
        self.hop_length = 64

        self.conv_blocks = nn.Sequential(
            self._conv_block(1, 64, dropout=0.3),
            self._conv_block(64, 128, dropout=0.3),
            self._conv_block(128, 256, dropout=0.4),
            self._conv_block(256, 512, dropout=0.4),
            self._conv_block(512, 256, dropout=0.4)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Lazy initialized RNN and Attention
        self.rnn1 = self.rnn2 = None
        self.attention = None
        self.rnn_input_size = None

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )

    def _compute_stft_batch(self, x):
        """Compute log-magnitude STFT using torch.stft (B, T) → (B, 1, F, T)"""
        x = x.unsqueeze(1)  # (B, 1, T)
        window = torch.hann_window(window_length=self.n_fft).to(x.device)

        stft_result = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True
        )  # (B, F, T)

        mag = stft_result.abs()
        log_spec = torch.log1p(mag)
        return log_spec.unsqueeze(1)  # (B, 1, F, T)

    def forward(self, x, lengths=None):
        B = x.size(0)
        device = x.device

        # STFT → CNN → pool
        x = self._compute_stft_batch(x)
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)

        # Reshape for RNN
        x = x.permute(0, 3, 1, 2).contiguous().view(B, x.size(3), -1)

        if self.rnn1 is None:
            self.rnn_input_size = x.size(-1)
            self.rnn1 = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True).to(device)
            self.rnn2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True).to(device)
            self.attention = Attention(input_dim=512).to(device)
            print(f"Initialized RNNs and Attention with input size: {self.rnn_input_size}")

        x, _ = self.rnn1(x)
        x = self.dropout2(x)
        x, _ = self.rnn2(x)
        x, _ = self.attention(x)

        # FC classification
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)

        return self.fc4(x)

class ECGTrainer:
    def __init__(self, model, device='cpu', name_prefix=''):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.name_prefix = name_prefix

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets, lengths) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data, lengths)
            loss = self.criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets, lengths in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data, lengths)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')

        return total_loss / len(val_loader), accuracy, f1

    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_f1 = 0
        patience_counter, epoch = 0, 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                if os.path.exists('./Uni/AMLS'):
                    torch.save(self.model.state_dict(), './Uni/AMLS/' + self.name_prefix + self.model.__class__.__name__ + '_best_model.pth')
                else:
                    torch.save(self.model.state_dict(), self.name_prefix + self.model.__class__.__name__ + '_best_model.pth')

            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # Load best model
        if os.path.exists('./Uni/AMLS'):
            self.model.load_state_dict(torch.load('./Uni/AMLS/' + self.name_prefix + self.model.__class__.__name__  + '_best_model.pth'))
        else:
            self.model.load_state_dict(torch.load(self.name_prefix + self.model.__class__.__name__  + '_best_model.pth'))

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_f1': best_f1
        }

        

def generate_base_predictions(model, test_loader, device, pre=''):
  model.eval()
  base_predictions = []
  with torch.no_grad():
      for data, _, lengths in test_loader:
          data = data.to(device)
          outputs = model(data, lengths)
          _, predicted = outputs.max(1)
          base_predictions.extend(predicted.cpu().numpy())

  # Save base predictions
  adr = pre + model.__class__.__name__ + '_base.csv'
  if os.path.exists('./Uni/AMLS'):
    adr = './Uni/AMLS/' + pre + model.__class__.__name__ + '_base.csv'
  pd.DataFrame({'label': base_predictions}).to_csv(adr, index=False)
  print(f"Base predictions saved to {adr}")
  
train_dataset = ECGDataset(X_train_split, y_train_split, is_training=True)
val_dataset = ECGDataset(X_val_split, y_val_split, is_training=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# TASK 2: MODELING AND TUNING 
# MODEL 1: SimpleSTFTModel
print("\n" + "TASK 2: MODELING AND TUNING ")

print("Training SimpleSTFT...")

simple_model = SimpleSTFTModel(num_classes=4)
trainer = ECGTrainer(simple_model, device=device)

print(f'simple_model initialized with {sum(p.numel() for p in simple_model.parameters())} parameters')
print("Starting training...")

history = trainer.train(train_loader, val_loader, epochs=70, patience=40)

test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
generate_base_predictions(simple_model, test_loader, device)

plot_training_history(history, model_name="SimpleSTFTModel", save_path="training_simple_model_history.png")


# MODEL 2: ImprovedSTFTModel
print("\n" + "TASK 2: MODELING AND TUNING ")
print("Training ImprovedSTFT...")

improved_model = ImprovedSTFTModel(num_classes=4)
trainer = ECGTrainer(improved_model, device=device)

print(f'Model initialized with {sum(p.numel() for p in improved_model.parameters())} parameters')
print("Starting training...")

history_improved = trainer.train(train_loader, val_loader, epochs=90, patience=40)
test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
generate_base_predictions(improved_model, test_loader, device)
plot_training_history(history_improved, model_name="ImprovedSTFTModel", save_path="training_improve_model_history.png")

