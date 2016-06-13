/**
 * Break and axis at a specified location.
 *
 * Breaking an axis is useful when the axis scale covers a range that prevents detail in the figure being visible.
 *
 * @param
 * @return :
 */
 function break_axis(axisScale, domainLimits, rangeLimits, breakStart, breakEnd, rangeDivider=20, tickCount=10)
 {
    // Determine the upper and lower limits of the domain and range.
    var domainLower = domainLimits[0],
        domainUpper = domainLimits[1],
        rangeLower = rangeLimits[0],
        rangeUpper = rangeLimits[1];

    // Determine how to aportion the range between the three scale sections. The portion of the domain between the
    // break points will be forced to occupy (1 / rangeDivider) of the range, while the portions before and after
    // the break will occupy sections of the remaining available range proportional to the fraction of the domain
    // they cover.
    var domainLength = domainUpper - domainLower;
    var nonBreakDomainLength = domainLength - (breakEnd - breakStart);  // Domain not in the break zone.
    var preBreakDomainFraction = (breakStart - domainLower) / nonBreakDomainLength;  // Domain before the break.
    var nonBreakRange = (rangeUpper - rangeLower) * ((rangeDivider - 1) / rangeDivider);  // Range outside the break.
    var breakRange = [rangeLower + (nonBreakRange * preBreakDomainFraction),
                      rangeLower + ((preBreakDomainFraction + (1 / rangeDivider)) * nonBreakRange)];

    // Set the domain and range of the scale.
    axisScale
        .domain([domainLower, breakStart, breakEnd, domainUpper])
        .range([rangeLower, breakRange[0], breakRange[1], rangeUpper]);

    // Generate the ticks that should be used.
    var tickGenerator = d3.scale.linear()
        .domain([0, nonBreakDomainLength]);
    var ticks = tickGenerator.ticks(tickCount);
    ticks = ticks.map(function(currentValue)
        {
            if (currentValue - breakStart < 0)
            {
                return currentValue;
            }
            else if (currentValue - breakStart > 0)
            {
                return breakEnd + currentValue - breakStart;
            }
        });
    ticks = ticks.filter(function(n) { return n != undefined });

    return ticks;
 }