# Arrow-detection-bot

### Algorithm

The algorithm currently can determine only basic arrows
i.e is with 4 conesutive right angles with 2 acute angles bounding them

1. First the frame is converted to grayscale.
2. Then adaptive threshold is applied to determine the edges of shapes.
3. Then using findContours method contours are find.
4. Then for each contour it is approximated to a closed polygon.
5. Then a polygon with 7 sides is chosen using len(approx).
6. Then we find 4 consecutive right angles with 2 acute angles adjacent to the right angle.
7. This determine whether the shape is an arrow.
8. Then to determine the direction of the head of arrow and the tail is found and there slope is calculated.
9. For a given range of slope the arrows points right and for the some other range it points left.
