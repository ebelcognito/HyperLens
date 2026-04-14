#!/bin/bash

# Path to environments
DEEPLENS_ENV="../../../../playpen/hisham26/deeplens_env"
SIMU1_ENV="../../../../playpen/hisham26/simu1_env"
OPTIFUSION_ENV="../../../../playpen/hisham26/optifusion_env"

# Function to activate DeepLens environment
activate_deeplens() {
    echo "Activating DeepLens environment..."
    source $DEEPLENS_ENV/bin/activate
    echo "DeepLens environment activated!"
}

# Function to activate Simu1 environment
activate_simu1() {
    echo "Activating Simu1 environment..."
    source $SIMU1_ENV/bin/activate
    echo "Simu1 environment activated!"
}

# Function to activate OptiFusion environment
activate_optifusion() {
    echo "Activating OptiFusion environment..."
    source $OPTIFUSION_ENV/bin/activate
    echo "OptiFusion environment activated!"
}

# Default to DeepLens environment if no argument is provided
if [ "$1" == "simu1" ]; then
    activate_simu1
elif [ "$1" == "optifusion" ]; then
    activate_optifusion
else
    activate_deeplens
fi

# Print usage information
echo ""
echo "Environment paths:"
echo "DeepLens: $DEEPLENS_ENV"
echo "Simu1: $SIMU1_ENV"
echo ""
echo "Usage:"
echo "  source activate_env.sh            # Activate DeepLens environment"
echo "  source activate_env.sh simu1      # Activate Simu1 environment"
echo "  source activate_env.sh optifusion # Activate OptiFusion environment"
