function [target_id] = get_target_id(reachingAngle)
    if (reachingAngle>=10) & (reachingAngle<50)
        target_id = 1;
    elseif (reachingAngle>=50) & (reachingAngle<90)
        target_id = 2;
    elseif (reachingAngle>=90) & (reachingAngle<130)
        target_id = 3;
    elseif (reachingAngle>=130) & (reachingAngle<170)
        target_id = 4;
    elseif (reachingAngle>=170) & (reachingAngle<210)
        target_id = 5;
    elseif (reachingAngle>=210) & (reachingAngle<270)
        target_id = 6;
    elseif (reachingAngle>=270) & (reachingAngle<330)
        target_id = 7;
    elseif (reachingAngle>=330) & (reachingAngle<360)
        target_id = 8;
    elseif (reachingAngle<10)
        target_id = 8;
    end
end