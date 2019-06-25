# -*- coding: utf-8 -*-
"""Histogram.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o0Lm5oXyOkSpN9tPNHFPVg-Jyke7ie9g
"""

# tf.summary. Histogram

import tensorflow as tf

tf.reset_default_graph()

s_scalar=tf.get_variable(name='s_scalar',shape=[],initializer=tf.truncated_normal_initializer(mean=0,stddev=1))

y_matrix=tf.get_variable(name='y_matrix',shape=[20,30],initializer=tf.truncated_normal_initializer(mean=5,stddev=1))

first_summary=tf.summary.scalar(name='first_summary',tensor=s_scalar)

histogram_summary=tf.summary.histogram('histogram_summary', y_matrix)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    writer=tf.summary.FileWriter('./log',sess.graph)
    for i in range(100):
        sess.run(init)
        summary1=sess.run(histogram_summary)
        summary2=sess.run(first_summary)     
        writer.add_summary(summary1,i)
        writer.add_summary(summary2,i)

