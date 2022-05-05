from phantom import phantom
import matplotlib.pyplot as plt

shepp_logan = phantom(n = 512, p_type = 'modified shepp-logan', ellipses = None)
plt.imshow(shepp_logan)
plt.gray()
plt.show()