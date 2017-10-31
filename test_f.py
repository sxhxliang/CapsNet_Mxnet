import utils


train_data, test_data = utils.load_data_mnist(batch_size=2,resize=28)

for i, batch in enumerate(train_data):
	data, label = batch

	print('data, label',data, label)