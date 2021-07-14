from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from sklearn.metrics import classification_report


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    name_label = dataset.dataset.classes
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, pre_, label_origin = model.test()
        writer.update_counter(ncorrect, nexamples)
        print(classification_report(label_origin, pre_, target_names=name_label))
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
