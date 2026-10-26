# Randomized Tensor Compression CXX

## Project context

This is a final project for a parallel programming course. The goal is
to compare the performance of C++ implementations of the Star-M SVD 1
and 2 algorithms to the current MATLAB implementation. Specifically
targetting a machine equipped with a 32 core AMD-7975WX with 
Simultaneous Multi-threading disabled, 8X16GB DDR5 memory, no overclocking,
running Windows 11 Pro.

## Coding rules

- **No `new`/`delete`.** Use smart pointers (`std::unique_ptr`,
  `std::shared_ptr`, etc) for all owned allocations.
- **Prefer specified width types.** Rather than use int, short, long, etc,
  prefer their equivalents with specified widths, int8_t, uint32_t, etc.
- **No platform-specific threading APIs.** Use `std::thread`/`std::jthread`
  (and the rest of `<thread>`/`<stop_token>`), not pthreads/Win32 threads.

## Style guide

### Function and block comments

- **Comment all code.** Every function, class, struct, etc MUST have a comment
  describing EVERY parameter, return value, member variable, exceptions thrown, etc.
  There are no exceptions to this rule. Even something as simple as deleting
  the move constructor should have a comment describing why it is deleted.
  Furthermore, any non-trivial code block should have a comment describing it.
  All comments should focus less on the actual code implemented, and more on how
  the code fits into the bigger picture/the intent of the code.
  Comments should be written in the header file except for static functions,
  which should be commented in the source file.

A deleted constructor needs a comment explaining the restriction. For example,
a worker driver whose threads capture `this` must keep a stable address:

```cpp
/**
 * Disables move construction because worker threads capture this driver's
 * address. Transferring its state to another instance would leave those
 * threads referring to the original object.
 *
 * @param other The driver that would otherwise supply the moved state.
 */
AsyncParallelForDriver(AsyncParallelForDriver &&other) = delete;
```

Document inputs and results next to the function declaration. For example:

```cpp
/**
 * Converts a rectangle from world coordinates to screen coordinates.
 *
 * @param world The rectangle in world coordinates.
 *
 * @return The rectangle in screen coordinates.
 */
Rectangle2D WorldToScreen(Rectangle2D world) const;
```

Describe preconditions and applicable exceptions as part of the function's
contract. Explain the purpose of the operation as well as its inputs:

```cpp
/**
 * Sets the zoom factor of the camera, keeping the specified screen
 * coordinates anchored to the same world coordinates.
 *
 * @param new_zoom The new zoom factor to set. Must be finite and greater
 * than zero.
 * @param screen_position The position in screen space to anchor.
 *
 * @throws std::invalid_argument if the new zoom factor is not finite or is
 * less than or equal to zero.
 */
void SetZoomAt(float new_zoom, Vector2D screen_position);
```

Document source-local static helpers in the source file. Comments within
non-trivial blocks explain the reasoning:

```cpp
/**
 * Checks if a circle is outside the bounds of the rendering output.
 *
 * @param circle The circle to check, in screen coordinates.
 * @param output_width The width of the rendering output.
 * @param output_height The height of the rendering output.
 *
 * @return true if the circle is outside the bounds of the rendering output,
 * false otherwise.
 */
static bool IsOutsideOutput(const Circle2D &circle, std::int32_t output_width,
                            std::int32_t output_height) {
    // Find the closest point on the rectangle defined by the output dimensions
    // to the center of the circle.
    const float closest_x =
        std::clamp(circle.x, 0.0F, static_cast<float>(output_width));
    const float closest_y =
        std::clamp(circle.y, 0.0F, static_cast<float>(output_height));

    // If the distance from the circle's center to this closest point is greater
    // than or equal to the radius, then the circle is outside the bounds of the
    // output.
    return std::hypot(circle.x - closest_x, circle.y - closest_y) >=
           circle.radius;
}
```

Include ownership, units, defaults, state changes, failure behavior, and reasons
for deleted operations when relevant. Keep comments accurate as the code changes.

### Struct comments

- All structs must document the exact field names. This is done to allow things like intellisense
  to report the field names as part of the documentation for a struct.

The struct-level comment explains the type's purpose and includes a `FIELDS:`
list using the exact identifiers from the declaration:

```cpp
/**
 * Represents a color with red, green, blue, and alpha components.
 * Each component is an 8-bit unsigned integer (0-255).
 * The default color is black with full opacity (alpha = 255).
 *
 * FIELDS:
 * - red: The red component of the color.
 * - green: The green component of the color.
 * - blue: The blue component of the color.
 * - alpha: The opacity component of the color.
 */
struct Color {
    std::uint8_t red = 0;
    std::uint8_t green = 0;
    std::uint8_t blue = 0;
    std::uint8_t alpha = 255;
};
```

### Enum comments

- Likewise, enums (and enum classes, and enum structs) must document the exact names of the enum values.

Use the exact enum value names in the enclosing documentation and explain the
behavior each selects:

```cpp
/**
 * Controls how entity sizes and positions are interpreted relative to the
 * current window size.
 *
 * MEMBERS:
 * - Constant: Pixel values are used verbatim, regardless of window size.
 * - Proportional: Pixel values are rescaled so their proportion of the
 *   screen stays constant across window sizes.
 */
enum class ScaleMode : std::uint8_t {
    Constant,
    Proportional,
};
```

### Code conventions

- **No trailing underscore on private member variables.** e.g. use
  `count`, not `count_`.
- If statements and else statements should always use curly brackets even if they are not necessary.
  And for loops and... etc. Blocks should be easily identifiable and not hidden in a single line.
- When writing code, do not try to silently succeed and ignore non-critical errors.
  Crashing with an informative log message is preferable to succeeding in a mysterious way,
  as such successes can lead to mysterious and difficult to debug issues.
- Use the general written code formatting found in `.clang-format` for formatting code.
  Run `git clang-format` when making changes to code to ensure that the formatting is correct.

### Complexity

Additionally, a core idea here is that we should earn our complexity.
For example, don't just slap a `[[nodiscard]]` attribute on a function
because "it's good practice elsewhere".
It's not that we don't care about fancy elements
of the C++ language, but instead that if we don't need to do something
in a complex way, we shouldn't. More than just about anything else,
this should guide everyone and everything developing in this project.
Eventually, maybe we will have a reason to use `[[nodiscard]]`.
But until a clear pattern in which some feature should be used emerges,
we should avoid it.
Or templates. We don't need a template for something only used once.
But if we do find ourselves using a common function with just a difference
in type that implements essentially the same logic, then we must refactor
or write a template. But until that point, we should avoid it.