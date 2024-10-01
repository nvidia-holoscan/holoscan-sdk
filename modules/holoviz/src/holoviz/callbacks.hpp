/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_CALLBACKS_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_CALLBACKS_HPP

#include <cstdint>

namespace holoscan::viz {

/**
 * Keys codes. Printable keys use the ASCII value, function keys are put in the 256+ range.
 *
 * The key codes match the GLFW key codes https://www.glfw.org/docs/latest/group__keys.html.
 */
enum class Key {
  // Printable keys
  SPACE = int(' '),
  APOSTROPHE = int('\''),
  COMMA = int(','),
  MINUS = int('-'),
  PERIOD = int('.'),
  SLASH = int('/'),
  ZERO = int('0'),
  ONE = int('1'),
  TWO = int('2'),
  THREE = int('3'),
  FOUR = int('4'),
  FIVE = int('5'),
  SIX = int('6'),
  SEVEN = int('7'),
  EIGHT = int('8'),
  NINE = int('9'),
  SEMICOLON = int(';'),
  EQUAL = int('='),
  A = int('A'),
  B = int('B'),
  C = int('C'),
  D = int('D'),
  E = int('E'),
  F = int('F'),
  G = int('G'),
  H = int('H'),
  I = int('I'),
  J = int('J'),
  K = int('K'),
  L = int('L'),
  M = int('M'),
  N = int('N'),
  O = int('O'),
  P = int('P'),
  Q = int('Q'),
  R = int('R'),
  S = int('S'),
  T = int('T'),
  U = int('U'),
  V = int('V'),
  W = int('W'),
  X = int('X'),
  Y = int('Y'),
  Z = int('X'),
  LEFT_BRACKET = int('['),
  BACKSLASH = int('\\'),
  RIGHT_BRACKET = int(']'),
  GRAVE_ACCENT = int('`'),
  // function keys
  ESCAPE = 256,
  ENTER = 257,
  TAB = 258,
  BACKSPACE = 259,
  INSERT = 260,
  DELETE = 261,
  RIGHT = 262,
  LEFT = 263,
  DOWN = 264,
  UP = 265,
  PAGE_UP = 266,
  PAGE_DOWN = 267,
  HOME = 268,
  END = 269,
  CAPS_LOCK = 280,
  SCROLL_LOCK = 281,
  NUM_LOCK = 282,
  PRINT_SCREEN = 283,
  PAUSE = 284,
  F1 = 290,
  F2 = 291,
  F3 = 292,
  F4 = 293,
  F5 = 294,
  F6 = 295,
  F7 = 296,
  F8 = 297,
  F9 = 298,
  F10 = 299,
  F11 = 300,
  F12 = 301,
  F13 = 302,
  F14 = 303,
  F15 = 304,
  F16 = 305,
  F17 = 306,
  F18 = 307,
  F19 = 308,
  F20 = 309,
  F21 = 310,
  F22 = 311,
  F23 = 312,
  F24 = 313,
  F25 = 314,
  KP_0 = 320,
  KP_1 = 321,
  KP_2 = 322,
  KP_3 = 323,
  KP_4 = 324,
  KP_5 = 325,
  KP_6 = 326,
  KP_7 = 327,
  KP_8 = 328,
  KP_9 = 329,
  KP_DECIMAL = 330,
  KP_DIVIDE = 331,
  KP_MULTIPLY = 332,
  KP_SUBTRACT = 333,
  KP_ADD = 334,
  KP_ENTER = 335,
  KP_EQUAL = 336,
  LEFT_SHIFT = 340,
  LEFT_CONTROL = 341,
  LEFT_ALT = 342,
  LEFT_SUPER = 343,
  RIGHT_SHIFT = 344,
  RIGHT_CONTROL = 345,
  RIGHT_ALT = 346,
  RIGHT_SUPER = 347,
  MENU = 348
};

/// Key and mouse button actions
enum class KeyAndButtonAction {
  PRESS,    ///< the key or mouse button was pressed
  RELEASE,  ///< the key or mouse button was released
  REPEAT    ///< the key was held down until it repeated
};

/// Key modifiers
struct KeyModifiers {
  bool shift : 1;      ///< one or more shift keys where held down
  bool control : 1;    ///< one or more control keys where held down
  bool alt : 1;        ///< one or more alt keys where held down
  bool caps_lock : 1;  ///< caps lock key is enabled
  bool num_lock : 1;   ///< num lock key is enabled
};

/// Mouse buttons
enum class MouseButton {
  LEFT,    ///< left
  MIDDLE,  ///< middle
  RIGHT    ///< right
};

/**
 * Function pointer type for key callbacks.
 *
 * @param user_pointer user pointer value
 * @param key the key that was pressed
 * @param action key action (PRESS, RELEASE, REPEAT)
 * @param modifiers bit field describing which modifieres were held down
 */
typedef void (*KeyCallbackFunction)(void* user_pointer, Key key, KeyAndButtonAction action,
                                    KeyModifiers modifiers);

/**
 * Function pointer type for Unicode character callbacks.
 *
 * @param user_pointer user pointer value
 * @param code_point Unicode code point of the character
 */
typedef void (*UnicodeCharCallbackFunction)(void* user_pointer, uint32_t code_point);

/**
 * Function pointer type for mouse button callbacks.
 *
 * @param user_pointer user pointer value
 * @param button the mouse button that was pressed
 * @param action button action (PRESS, RELEASE)
 * @param modifiers bit field describing which modifieres were held down
 */
typedef void (*MouseButtonCallbackFunction)(void* user_pointer, MouseButton button,
                                            KeyAndButtonAction action, KeyModifiers modifiers);

/**
 * Function pointer type for scroll callbacks.
 *
 * @param user_pointer user pointer value
 * @param x_offset scroll offset along the x-axis
 * @param y_offset scroll offset along the y-axis
 */
typedef void (*ScrollCallbackFunction)(void* user_pointer, double x_offset, double y_offset);

/**
 * Function pointer type for cursor position callbacks.
 *
 * @param user_pointer user pointer value
 * @param x_pos new cursor x-coordinate in screen coordinates, relative to the left edge of the
 * content area
 * @param y_pos new cursor y-coordinate in screen coordinates, relative to the left edge of the
 * content area
 */
typedef void (*CursorPosCallbackFunction)(void* user_pointer, double x_pos, double y_pos);

/**
 * Function pointer type for framebuffer size callbacks.
 *
 * @param user_pointer user pointer value
 * @param width new width of the framebuffer in pixels
 * @param height new height of the framebuffer in pixels
 */
typedef void (*FramebufferSizeCallbackFunction)(void* user_pointer, int width, int height);

/**
 * Function pointer type for window size callbacks.
 *
 * @param user_pointer user pointer value
 * @param width new width of the window in screen coordinates
 * @param height new height of the window in screen coordinates
 */
typedef void (*WindowSizeCallbackFunction)(void* user_pointer, int width, int height);

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_CALLBACKS_HPP */
