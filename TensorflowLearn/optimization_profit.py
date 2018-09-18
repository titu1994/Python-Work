import tensorflow as tf
tf.enable_eager_execution()

# Available Flour = 30
# Available Eggs = 40

# Pasta recipe => 1 Pasta = 2.5 * Flour + 5 * Eggs
# Bread recipe => 1 Bread = 3.5 * Flour + 2.5 * Eggs

# Pasta Sale Price = 3
# Bread Sale Price = 2.5

# Constraints
# 2.5 Pasta + 3.0 Bread <= 30 # Flour
# 5.0 Pasta + 2.5 Bread <= 40 # Eggs
# Bread >= 0
# Pasta >= 0

# Maximize : 3 * Pasta + 2.5 * Bread

# Use non neg constraint to force last 2 constrains
pasta_t = tf.get_variable('pasta', initializer=0., constraint=tf.keras.constraints.non_neg())
bread_t = tf.get_variable('breads', initializer=0., constraint=tf.keras.constraints.non_neg())


# Flour cost (per pizza and per bread)
def flour():
    res = 2.5 * pasta_t + 3.5 * bread_t
    return res

# Eggs cost (per pizza and per bread)
def eggs():
    res = 5.0 * pasta_t + 2.5 * bread_t
    return res

# Profit per pizza and bread
def profit():
    return 3.0 * pasta_t + 2.5 * bread_t

# Additional constraints on available flour and eggs
# Can substitute square instead of abs for smoother fit
def constraint():
    return tf.square(29.5 - flour()) + \
           tf.square(39.5 - eggs())

# Objective function - to be minimized (minimize constraints, maximize profits)
def objective():
    return constraint() - profit()


optimizer = tf.train.GradientDescentOptimizer(0.01)

for i in range(200):
    with tf.GradientTape() as tape:
        loss = objective()
    gradients = tape.gradient(loss, [pasta_t, bread_t])

    grad_vars = zip(gradients, [pasta_t, bread_t])
    optimizer.apply_gradients(grad_vars, global_step=tf.train.get_or_create_global_step())

    print("Objective : ", objective().numpy())
    print("Profit : ", profit().numpy())
    print()

p = pasta_t.numpy()
b = bread_t.numpy()
print("Pasta : ", p, "Bread : ", b)
print("Flour : ", flour().numpy(), "Eggs : ", eggs().numpy())
