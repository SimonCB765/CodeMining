/**
 * Create a legend where the glyphs are paths.
 *
 * @param figureContainer:  The d3 selection of the parent element for the legend. The legend will be added under it
 *                              in the DOM.
 * @param labels:           An array of the labels for the items in the figure. Will be used as the legend text.
 * @param legendXLoc:       The X coordinate of the top left corner of the legend border.
 * @param legendYLoc:       The Y coordinate of the top left corner of the legend border.
 * @param itemPadding:      The padding between the legend border and the rows of items, the labels and the glyphs
 *                              within a row and between each row of items.
 * @param pathLength:       The Euclidean distance between the path start and end.
 * @return :                The d3 selection of the legend container.
 */
function create_path_legend(figureContainer, labels, legendXLoc, legendYLoc, itemPadding=10, pathLength=50)
{
    // Create the legend container.
	var legend = figureContainer.append("g")
	    .classed("legend", true)
	    .attr("transform", "translate(" + legendXLoc + "," + legendYLoc + ")");

    // Create the container for the legend items.
    var legendItems = legend.append("g")
        .classed("legend-items", true);

    // Create the rows of legends and glyphs.
    var cumulativeItemHeight = 0;  // The cumulative height of the rows of items (plus the padding in between the rows).
    var maxItemWidth = 0;  // The width of the widest label/glyph rows (used to determine borer width).
	for (i = 0; i < labels.length; i++)
	{
	    // Add the label on this row.
	    var label = legendItems.append("text")
	        .classed("legend-label", true)
            .attr("x", pathLength + itemPadding)  // X position is itemPadding pixels away from the end of the path.
            .attr("y", cumulativeItemHeight)  // Y position is relative to the preceding labels.
            .text(labels[i]);
        var labelSize = label.node().getBBox();  // Bounding box for the label.

        // Add the glyph. This is just a curved path. The length of the path is not taken to be the curved length, but
        // rather the direct Euclidean distance between the start and end points.
        var glyphStart = {"x": 0, "y": labelSize.y + (0.5 * labelSize.height)};
        var glyphControl1 = {"x": glyphStart.x + (pathLength / 3), "y": labelSize.y};
        var glyphControl2 = {"x": glyphStart.x + (pathLength * 2 / 3), "y": labelSize.y + labelSize.height};
        var glyphEnd = {"x": glyphStart.x + pathLength, "y": glyphStart.y};
        var glyphPath = "M" + glyphStart.x + "," + glyphStart.y + "C" + glyphControl1.x + "," + glyphControl1.y + "," +
            glyphControl2.x + "," + glyphControl2.y + "," + glyphEnd.x + "," + glyphEnd.y;
        var glyph = legendItems.append("path")
            .classed("legend-glyphPath", true)
            .classed(labels[i].replace(/ /g, "_"), true)  // Replace spaces in the label names to make the class work.
            .attr("d", glyphPath);

        // See if this row is wider than the previous widest found and add its height to the cumulative total height.
        maxItemWidth = Math.max(maxItemWidth, labelSize.width + itemPadding + glyph.node().getBBox().width)
        cumulativeItemHeight += itemPadding + labelSize.height;
	}

    // Reposition the label items so that they will sit in the middle of the border (as the border is fixed to have
    // a specific top left corner position by the function arguments). The X position is simply shifted by the
    // desired padding. However, the Y position may need to be altered slightly to prevent the text going higher than
    // desired. The items may therefore have to be shifted down farther than the item padding dictates in order to get
    // a uniform padding between the rows and the border.
	legendItems
        .attr("transform", "translate(" + itemPadding + "," + (itemPadding - legendItems.node().getBBox().y)+ ")")

	// Create the legend border.
	legend.append("rect")
	    .classed("legend-border", true)
	    .attr("x", 0)
	    .attr("y", 0)
	    .attr("height", legendItems.node().getBBox().height + (itemPadding * 2))  // Height depends on number of rows.
	    .attr("width", maxItemWidth + (2 * itemPadding));  // Width dictated by widest combination of glyph and label.

	return legend
}

/**
 * Create a legend where the glyphs are squares.
 *
 * @param figureContainer:  The d3 selection of the parent element for the legend. The legend will be added under it
 *                              in the DOM.
 * @param labels:           An array of the labels for the items in the figure. Will be used as the legend text.
 * @param legendXLoc:       The X coordinate of the top left corner of the legend border.
 * @param legendYLoc:       The Y coordinate of the top left corner of the legend border.
 * @param itemPadding:      The padding between the legend border and the rows of items, the labels and the glyphs
 *                              within a row and between each row of items.
 * @return :                The d3 selection of the legend container.
 */
function create_square_legend(figureContainer, labels, legendXLoc, legendYLoc, itemPadding=10)
{
    // Create the legend container.
	var legend = figureContainer.append("g")
	    .classed("legend", true)
	    .attr("transform", "translate(" + legendXLoc + "," + legendYLoc + ")");

    // Create the container for the legend items.
    var legendItems = legend.append("g")
        .classed("legend-items", true);

    // Create the rows of legends and glyphs.
    var cumulativeItemHeight = 0;  // The cumulative height of the rows of items (plus the padding in between the rows).
    var maxItemWidth = 0;  // The width of the widest label/glyph rows (used to determine borer width).
	for (i = 0; i < labels.length; i++)
	{
	    // Add the label on this row.
	    var label = legendItems.append("text")
	        .classed("legend-label", true)
            .attr("x", 0)  // X position is set once the glyph is created..
            .attr("y", cumulativeItemHeight)  // Y position is relative to the preceding labels.
            .text(labels[i]);
        var labelSize = label.node().getBBox();  // Bounding box for the label.

        // Add the glyph.
        var glyphShrinkage = 4;  // Pixels to shrink the glyph by to account for the text not filling its height.
        var glyph = legendItems.append("rect")
            .classed("legend-glyphSquare", true)
            .classed(labels[i].replace(/ /g, "_"), true)  // Replace spaces in the label names to make the class work.
            .attr("x", 0)
            .attr("y", labelSize.y + (glyphShrinkage / 2))
            .attr("width", labelSize.height - glyphShrinkage)
            .attr("height", labelSize.height - glyphShrinkage);

        // Reset the label's X position relative to the glyph width. Must be done after the glyph is added as you
        // don't know the width of the glyph until the text is created.
        label.attr("x", labelSize.height + itemPadding)

        // See if this row is wider than the previous widest found and add its height to the cumulative total height.
        maxItemWidth = Math.max(maxItemWidth, labelSize.width + itemPadding + glyph.node().getBBox().width)
        cumulativeItemHeight += itemPadding + labelSize.height;
	}

    // Reposition the label items so that they will sit in the middle of the border (as the border is fixed to have
    // a specific top left corner position by the function arguments). The X position is simply shifted by the
    // desired padding. However, the Y position may need to be altered slightly to prevent the text going higher than
    // desired. The items may therefore have to be shifted down farther than the item padding dictates in order to get
    // a uniform padding between the rows and the border.
	legendItems
        .attr("transform", "translate(" + itemPadding + "," + (itemPadding - legendItems.node().getBBox().y)+ ")")

	// Create the legend border.
	legend.append("rect")
	    .classed("legend-border", true)
	    .attr("x", 0)
	    .attr("y", 0)
	    .attr("height", legendItems.node().getBBox().height + (itemPadding * 2))  // Height depends on number of rows.
	    .attr("width", maxItemWidth + (2 * itemPadding));  // Width dictated by widest combination of glyph and label.

	return legend
}