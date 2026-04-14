Brief summary of what we've been doing

- Iniitally, we created the environment
    Challenges: finding an optimal 2D environment
    Needed to open source python based envs for free form optics
    Ray-Optics provided it but it was based on node.js
    This was only web based -> TOOK 1 SECOND PER REQUEST
    Brought this down by making the node instance persistent
    Then we implemented a bunch of unit tests to see that our Python bridge was working fine
        (Ray shooting, detector power, effect of a quad moving in path)
    Then we created a simple aligning env (single ray, flat surfaces)
    But then realised that there are only two possible detector powers in this!
    We then tried it when the source and detector where having a random initial state
        But this again has just 2 solutions!
    We tried to vary both the x and y of the quad -> not sure what the problem with this was...
        This solution space is again quite sparse, can be deterministically solved.. not very interesting
    
    We just jumped to adding curved surfaces to it as well
    Initiall, sag was fixed while y varied
    This learned an env which had the quad to just keep moving down
    tried entropy reward, didn't work
    Tried to filter the env to only have cases where the power was less than 0.5 in the beginning
    Still didn't yield much
    now trying 

        sag: in control param, boundary loss, boundary loss for both sag and y actions, and patience Termination

        Now I added a new thing, if the power is good enough (0.90) consecutively, then just supress this early termination thing

        Also MLP is 256, 256 instead of 64,64

    
Alright time for a zoom out
- Blind chicken work will kill you, take a step back and think exactly what is going on
- this is using an on-policy method to get an MLP to learn how to control this complex lens design framework


Questions worth exploring:
- Is a simple MLP enough?
Look at previous works that use it for photonic engineering

- How can we speed up training? Parallel processing
This is like 1-2 hour task -> promptable

- What are mistakes that cause this model to stagnate, freeze? How are the actions really being decided?
track the action updates in a better way and do ablations

- Is this representation truly the best one to showcase "freeform optics" in the wild?
mostly yes rn

- What exactly would a differentiable ray tracing framework give us? how would the loss gradient over detector help? right now, it is a simple linear metric, PSF i.e loop over f ( x, y ) would require a function to be defined
When sharpness becomes the metric it would absolutely be important to use the dO framework
1-2 hour work -> promptable

- Would PSF definition really be required or would it be enough to keep the detector small?






    
