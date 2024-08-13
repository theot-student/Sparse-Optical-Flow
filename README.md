# Sparse-Optical-Flow

Pour utiliser la méthode de calcul du flot optique, il faut changer les différents paramètres dans calculOpticalFlow.py. Pour les paramètres, il y à :
* filename                   : nom de la vidéo
* path                       : chemin du répertoire où se trouve la vidéo
* weights                    : poids du model entre le terme TV-L1 et sparsity. La valeur est comprise entre [0, 1]. 0 = sparse; 1 = not sparse
*  regs                      : Poids du régularisateur. La valeur est comprise entre [0, 1]. 0 = régularisation forte
* gradient_step              : pas du gradient pour ADAM scheduler
* precision                  : stop precision pour l'optimisation
* isRGB                      : booleen True si l'image est RGB
* isRed                      : booleen True si l'image est seulement rouge. Si isRed et isRGB sont False l'algorithme comprends que c'est une image en grayscale
* nbOfImages                 : nombre d'image de la vidéo à traité. Si 0 l'algorithme traite toute la vidéo
* save_directory             : chemin du répertoire de sauvegarde
* method = 1                 : choix de la norme utilisé dans la méthode. 1 = l_1 ; 2 = l_2 ; 3 = mix l_1 l_2
* init = 0                   : initialisation du champ de vecteur. 0 = init at zero ; 1 = init at random between -1 and 1 ; 2 = init at random between -0.1 and 0.1
