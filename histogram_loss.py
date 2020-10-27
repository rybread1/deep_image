import tensorflow as tf

def hist_match(source, template):
    shape = tf.shape(source)
    print(source)
    source = tf.reshape(source, [-1])
    template = tf.reshape(template, [-1])
    # get the set of unique pixel values and their corresponding indices and
    # counts

    hist_bins = 255

    # Defining the 'x_axis' of the histogram

    max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
    min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])

    hist_delta = (max_value - min_value)/hist_bins

    # Getting the x-axis for each value
    hist_range = tf.range(min_value, max_value, hist_delta)
    # I don't want the bin values; instead, I want the average value of each bin, which is 
    # lower_value + hist_delta/2
    hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))

    # Now, making fixed width histograms on this hist_axis 

    s_hist = tf.histogram_fixed_width(source, 
                                      [min_value, max_value],
                                        nbins = hist_bins, 
                                      dtype = tf.int64
                                      )


    t_hist = tf.histogram_fixed_width(template, 
                                        [min_value, max_value],
                                        nbins = hist_bins, 
                                      dtype = tf.int64
                                      )

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = tf.cumsum(s_hist)
    s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
    s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))

    t_quantiles = tf.cumsum(t_hist)
    t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
    t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))


    nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), 
                                  s_quantiles, dtype = tf.int64)

    # Finding the correct s-histogram bin for every element in source
    s_bin_index = tf.cast(tf.divide(source, hist_delta), tf.int64)

    ## In the case where an activation function of 0-1 is used, then there might be some index exception errors. 
    ## This is to deal with those
    s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)

    # Matching it to the correct t-histogram bin, and then making it the correct shape again
    matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
    return tf.reshape(matched_to_t, shape)

def hist_loss(calculated, target):
    if isinstance(calculated, list):
        histogram = ([hist_match(calc, targ)
                   for calc,  targ in zip(calculated, target)])

        loss = sum([tf.math.reduce_sum(tf.keras.losses.MSE(calc,hist))
                   for calc,  hist in zip(calculated, histogram)])
    else: 
        histogram = hist_match(calculated, target)
        loss = tf.math.reduce_sum(tf.keras.losses.MSE(calculated,histogram))
    return loss
