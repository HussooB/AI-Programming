import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 # Extended to find the new peak for the larger model

# Initialize the NEW, larger model
model = DemoGPT(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5) # Gentler learning rate

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

    val_accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        print("✓ Saved best model")

# Load the absolute best model for final testing
print("\nLoading best model for final test...")
model.load_state_dict(torch.load("best_model.pt"))
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
