import os
import numpy as np
#import tensorflow as tf
from PIL import Image
from skimage import io, transform
import tensorflow as tf
import matplotlib.pyplot as plt
#tf.disable_eager_execution()
#tf.disable_v2_behavior()

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# 只显示 Error


# 读取图片
def read_img(train_path,val_path, w, h):

    imgs = []
    labels = []
    val_imgs=[]
    val_labels=[]

    print('Start read the image ...')

    for img in os.listdir(train_path):
        img_path=os.path.join(train_path,img)
        label=img.split('.')[0].split('_')[0]#.split('_')[0]
        #print(label)
        im=io.imread(img_path)
        imgs.append(im)
        labels.append(label)
    print(labels)


    for img in os.listdir(val_path):
        img_path=os.path.join(val_path,img)
        label=img.split('.')[0].split('_')[0]#.split('_')[0]
        #print(label)
        im=io.imread(img_path)
        val_imgs.append(im)
        val_labels.append(label)
    print(val_labels)
    print('Finished ...')
    return np.asarray(imgs, np.float32), np.asarray(labels,np.float32) ,np.asarray(val_imgs,np.float32),np.asarray(val_labels,np.float32)


# 构建网络
def buildCNN(w, h, c):
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


    # 第一个卷积层 + 池化层（(90,120)——>(45,60))
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层 + 池化层 ((45,60)->(22,30))
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层 + 池化层 ((22,30)->(11,15))
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层 + 池化层 ((11,15)->(5,7))
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 5 * 7 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=20,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    print(x, y_)
    return logits, x, y_


# 返回损失函数的值，准确值等参数
def accCNN(logits, y_):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, train_op, correct_prediction, acc


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def runable(x_train, y_train, train_op, loss, acc, x, y_, x_val, y_val):
    # 训练和测试数据，可将n_epoch设置更大一些
    n_epoch = 50
    batch_size = 64
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_path="D:/pycharm_1/face_pandeng/"
    for epoch in range(n_epoch):
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("train loss: %f" % (train_loss / n_batch))
        print("train acc: %f" % (train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("validation loss: %f" % (val_loss / n_batch))
        print("validation acc: %f" % (val_acc / n_batch))
        if(val_acc/n_batch>=0.6):
            saver.save(sess, model_path)
        print('*' * 50)
    sess.close()
    '''# 绘制曲线
   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(np.arange(n_epoch), acc, label="Accuracy")
    #
    lns2=ax2.plot(np.arrange(n_epoch),loss,label="Loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training accuracy')
    ax2.set_ylabel('training loss')
    #combine
    lns=lns1+lns2
    labels=['Accuracy','Loss']
    plt.legend(lns,labels,loc=7)
    plt.show()
    '''



if __name__ == '__main__':
    train_p = 'D:/pycharm_1/database/train_5358'
    val_p='D:/pycharm_1/database/val_1429'
    w = 120
    h = 90
    c = 3

    ratio = 0.8  # 选取训练集的比例
  #
    x_train, y_train , x_val, y_val = read_img(train_path=train_p,val_path=val_p, w=w, h=h)
    logits, x, y_ = buildCNN(w=w, h=h, c=c)


    loss, train_op, correct_prediction, acc = accCNN(logits=logits, y_=y_)

    runable(x_train=x_train, y_train=y_train, train_op=train_op, loss=loss,
            acc=acc, x=x, y_=y_, x_val=x_val, y_val=y_val)
