from utils.utils import get_val_data, perform_val

if __name__ == "__main__":
    TARGET = 'lfw,talfw,agedb_30,calfw_961,cplfw_92867'
    vers = get_val_data('/raid/Data/eval/', TARGET)
    MULTI_GPU = False
    EMBEDDING_SIZE = 512
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 10
    model_root = '/raid/yy/insightface_torch/results/IR-CosFace-casia/Backbone_IR_50_Epoch_71_Batch_136000_Time_2020-05-27-20-54_checkpoint.pth'
    BACKBONE = IR_50([112,112]),
    BACKBONE.load_state_dict(torch.load(model_root))
    for ver in vers:
        name, data_set, issame = ver
        accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU,
                                                                          DEVICE,
                                                                          EMBEDDING_SIZE,
                                                                          BATCH_SIZE,
                                                                          BACKBONE,
                                                                          data_set,
                                                                          issame)
        buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
        print('[%s][%d]XNorm: %1.5f' % (name, batch + 1, xnorm))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch + 1, accuracy, std))
        print('[%s][%d]Best-Threshold: %1.5f' % (name, batch + 1, best_threshold))