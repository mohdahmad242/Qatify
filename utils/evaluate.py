import torch
import time

# Evaluation function
def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            inference_times.append(end_time - start_time)
    
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")
    print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")

    return accuracy, avg_inference_time