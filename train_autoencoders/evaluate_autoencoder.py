#code to test trained autoencoder

#test reconstruction
import matplotlib.pyplot as plt

# Get a batch of test images
test_batch = next(iter(data_loader))
images = test_batch['image'].to(device)
with torch.no_grad():
    outputs = autoencoder(images).sample

# Convert to CPU and detach
images = images.cpu().numpy()
outputs = outputs.cpu().numpy()

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 8, figsize=(15, 4))
for i in range(8):
    axes[0, i].imshow(images[i, 0], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')
    
    axes[1, i].imshow(outputs[i, 0], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Reconstructed')
plt.show()


# test CustomVAEEncoder class to make sure it works. CustomVAEEncoder class code commented in train_autoencoderKL.py file and will be pasted into edm2 code
encoder = CustomVAEEncoder(
    autoencoder=autoencoder,
    raw_mean=raw_mean,
    raw_std=raw_std,
    final_mean=0,
    final_std=0.5,
    batch_size=8
)

# Encode and decode an image
with torch.no_grad():
    latent = encoder.encode(test_batch['image'])
    reconstructed = encoder.decode(latent)

# Plot original and reconstructed images
original_image = test_batch['image'][0, 0].cpu().numpy()
reconstructed_image = reconstructed[0, 0].cpu().numpy()

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed')
plt.axis('off')
plt.show()
